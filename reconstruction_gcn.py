import os
import numpy as np
import random

import pandas as pd
import torch
import torch.utils.data
from datetime import datetime
from torch import nn, optim
import torch.nn.functional as F
from models import MolecularVAE
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import joblib
import os
import numpy as np
import random
import torch
import torch.utils.data
from datetime import datetime
from torch import nn, optim
import torch.nn.functional as F
from models import MolecularVAE
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import joblib
from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
DATA_PATH = './data/preprocessed'
with open('conf.yml') as f:
    cfg = load(f.read(), Loader=Loader)

device = cfg['model']['device']
model = torch.load('ae_graph_model.pth').to(device)
model.eval()

SEQ_DATA_P = './data/preprocessed'
GRAPH_DATA_P = './data/graph_tensors.npz'
with open('conf.yml') as f:
    cfg = load(f.read(), Loader=Loader)

alphabet_size = cfg['data']['alphabet_size']
seq_size = cfg['data']['seq_size']
node2idx = joblib.load(os.path.join(SEQ_DATA_P, 'node2idx.pkl'))
idx2node = joblib.load(os.path.join(SEQ_DATA_P, 'idx2node.pkl'))
test_seq = torch.Tensor(joblib.load(os.path.join(SEQ_DATA_P, 'test.pkl')))
with np.load(GRAPH_DATA_P) as loader:
    train_graph = torch.Tensor(loader['arr_1'])
    max_tensor = train_graph.max(dim=1)[0].max(dim=0)[0].unsqueeze(0).unsqueeze(0)
    test_graph = torch.Tensor(loader['arr_3'])
    test_graph = torch.nan_to_num(test_graph / max_tensor.expand(*test_graph.shape))
    test_graph = torch.cat(
        [torch.Tensor(loader['arr_2']), test_graph], dim=2)

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
test_loader = get_dataloader(test_seq, test_graph, batch_size=cfg['generation']['batch_size'])

true_list, pred_list = [], []

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD if cfg['model']['noise'] else BCE

for batch_idx, (seq_data, graph_data) in tqdm(enumerate(test_loader)):
    seq_data = torch.squeeze(seq_data).to(device)
    graph_data = torch.squeeze(graph_data).to(device)
    data_batch = seq_data.transpose(1, 2)
    recon_batch, mu, logvar = model(data_batch, graph_data)
    loss = loss_function(recon_batch, data_batch.transpose(1, 2), mu, logvar).item()
    loss /= cfg['generation']['batch_size'] * seq_size
    correct = (recon_batch.argmax(2) == torch.argmax(data_batch.transpose(1, 2), dim=2)).type(torch.float).sum().item()
    correct /= cfg['generation']['batch_size'] * seq_size
    pred_batch = recon_batch.swapdims(1, 2).argmax(1)
    for idx in range(cfg['generation']['batch_size']):
        true = ''.join([idx2node[i] for i in data_batch[idx, :, :].argmax(0).detach().cpu().numpy()])
        pred = ''.join([idx2node[i] for i in pred_batch[idx, :].detach().cpu().numpy()])
        true_list.append(true)
        pred_list.append(pred)
    break

result = pd.DataFrame(index=range(cfg['generation']['batch_size']),
                      columns=['smiles', 'reconstructed', 'loss', 'accuracy'])
result['smiles'] = true_list
result['reconstructed'] = pred_list
result['loss'] = loss
result['accuracy'] = correct
result.to_csv('ae_graph_sample.csv', sep='\t', index=False)