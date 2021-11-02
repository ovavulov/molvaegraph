import os
import numpy as np
import random
import torch
import torch.utils.data
from datetime import datetime
from torch import nn, optim
import torch.nn.functional as F
from models_gcn import MolecularGraphVAE
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import joblib

from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

with open('conf.yml') as f:
    cfg = load(f.read(), Loader=Loader)

start = datetime.now()

SEED = 19
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
warnings.filterwarnings('ignore')

SEQ_DATA_P = './data/preprocessed'
GRAPH_DATA_P = './data/graph_tensors.npz'
with open('conf.yml') as f:
    cfg = load(f.read(), Loader=Loader)

alphabet_size = cfg['data']['alphabet_size']
seq_size = cfg['data']['seq_size']
node2idx = joblib.load(os.path.join(SEQ_DATA_P, 'node2idx.pkl'))
idx2node = joblib.load(os.path.join(SEQ_DATA_P, 'idx2node.pkl'))
train_seq = torch.Tensor(joblib.load(os.path.join(SEQ_DATA_P, 'train.pkl')))
test_seq = torch.Tensor(joblib.load(os.path.join(SEQ_DATA_P, 'test.pkl')))
with np.load(GRAPH_DATA_P) as loader:
    train_graph = torch.Tensor(loader['arr_1'])
    max_tensor = train_graph.max(dim=1)[0].max(dim=0)[0].unsqueeze(0).unsqueeze(0)
    train_graph = torch.nan_to_num(train_graph / max_tensor.expand(*train_graph.shape))
    train_graph = torch.cat(
        [torch.Tensor(loader['arr_0']), train_graph], dim=2)
    test_graph = torch.Tensor(loader['arr_3'])
    test_graph = torch.nan_to_num(test_graph / max_tensor.expand(*test_graph.shape))
    test_graph = torch.cat(
        [torch.Tensor(loader['arr_2']), test_graph], dim=2)
tr_frac = cfg['data']['train_sample']
if tr_frac is not None:
    idxs = np.random.randint(0, len(train_seq), size=(int(len(train_seq) * tr_frac), ))
    train_seq = train_seq[idxs, :, :]
    train_graph = train_graph[idxs, :, :]
print(f'Train size >>\tsequence: {train_seq.shape} | graph: {train_graph.shape}')
te_frac = cfg['data']['test_sample']
if te_frac is not None:
    idxs = np.random.randint(0, len(test_seq), size=(int(len(test_seq) * te_frac), ))
    test_seq = test_seq[idxs, :, :]
    test_graph = test_graph[idxs, :, :]
print(f'Test size >>>\tsequence: {test_seq.shape} | graph: {test_graph.shape}')

class SeqAndGraphDataset(Dataset):
    def __init__(self, seq_tensor, graph_tensor):
        super(SeqAndGraphDataset, self).__init__()
        assert seq_tensor.shape[0] == graph_tensor.shape[0]
        self.seq_tensor = seq_tensor
        self.graph_tensor = graph_tensor
    def __len__(self):
        return len(self.seq_tensor)
    def __getitem__(self, idx):
        return self.seq_tensor[[idx], :, :], self.graph_tensor[[idx], :, :]

def get_dataloader(seq_data, graph_data, batch_size):
    dataset = SeqAndGraphDataset(seq_data, graph_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
#
train_loader = get_dataloader(train_seq, train_graph, batch_size=cfg['model']['batch_size'])
test_loader = get_dataloader(test_seq, test_graph, batch_size=cfg['model']['batch_size'])
#
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD if cfg['model']['noise'] else BCE

epochs = cfg['model']['epochs']
device = cfg['model']['device']
model = MolecularGraphVAE(device).to(device)
optimizer = optim.Adam(model.parameters(),
                       lr=cfg['model']['optimizer']['learning_rate'],
                       weight_decay=cfg['model']['optimizer']['weight_decay'])

def train_loop():
    model.train()
    train_loss = 0
    for batch_idx, (seq_data, graph_data) in tqdm(enumerate(train_loader)):
        seq_data = torch.squeeze(seq_data).to(device)
        graph_data = torch.squeeze(graph_data).to(device)
        seq_data = seq_data.transpose(1, 2)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(seq_data, graph_data)
        loss = loss_function(recon_batch, seq_data.transpose(1, 2), mu, logvar)
        loss.backward()
        train_loss += loss
        optimizer.step()
        if batch_idx % cfg['model']['verbose'] == 0:
            loss, current = loss.item(), batch_idx * len(seq_data)
            print(f"loss: {loss / len(seq_data):>7f}  [{current:>5d}/{len(train_seq):>5d}]")
            idx = np.random.randint(len(seq_data))
            pred = recon_batch.swapdims(1, 2).argmax(1)
            true = ''.join([idx2node[i] for i in seq_data[idx, :, :].argmax(0).detach().cpu().numpy()])
            pred = ''.join([idx2node[i] for i in pred[idx, :].detach().cpu().numpy()])
            print(f'True:\t{true}')
            print(f'Pred:\t{pred}')
    return train_loss / len(train_loader.dataset)

def test_loop():
    model.eval()
    test_loss, correct = 0, 0
    results = {'recons': [], 'true': []}
    for batch_idx, (seq_data, graph_data) in tqdm(enumerate(test_loader)):
        seq_data = torch.squeeze(seq_data).to(device)
        graph_data = torch.squeeze(graph_data).to(device)
        seq_data = seq_data.transpose(1, 2)
        recon_batch, mu, logvar = model(seq_data, graph_data)
        results['recons'].append(recon_batch)
        results['true'].append(seq_data)
        test_loss += loss_function(recon_batch, seq_data.transpose(1, 2), mu, logvar).item()
        correct += (recon_batch.argmax(2) == torch.argmax(seq_data.transpose(1, 2), dim=2)).type(torch.float).sum().item()
    test_loss /= len(test_loader) * len(seq_data)
    correct /= len(test_loader.dataset) * seq_size
    # joblib.dump(results, f'./data/recons_{epoch}.pkl')
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss / len(test_loader.dataset)

for epoch in range(1, epochs + 1):
    print(f'\n\nEpoch {epoch}...')
    train_loss = train_loop()
    test_loss = test_loop()

torch.save(model, 'ae_graph_model.pth')

finish = datetime.now()
time_spent = finish - start
print(f'Total time spent: {time_spent}')