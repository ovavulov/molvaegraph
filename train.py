import os
import numpy as np
import random
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

DATA_PATH = './data/preprocessed'
with open('conf.yml') as f:
    cfg = load(f.read(), Loader=Loader)

alphabet_size = cfg['data']['alphabet_size']
seq_size = cfg['data']['seq_size']
node2idx = joblib.load(os.path.join(DATA_PATH, 'node2idx.pkl'))
idx2node = joblib.load(os.path.join(DATA_PATH, 'idx2node.pkl'))
train = torch.Tensor(joblib.load(os.path.join(DATA_PATH, 'train.pkl')))
if cfg['data']['scaffolds']:
    test = torch.Tensor(joblib.load(os.path.join(DATA_PATH, 'test_scaffolds.pkl')))
else:
    test = torch.Tensor(joblib.load(os.path.join(DATA_PATH, 'test.pkl')))
tr_frac = cfg['data']['train_sample']
if tr_frac is not None:
    idxs = np.random.randint(0, len(train), size=(int(len(train) * tr_frac), ))
    train = train[idxs, :, :]
print('Train size:\t', train.shape)
te_frac = cfg['data']['test_sample']
if te_frac is not None:
    idxs = np.random.randint(0, len(test), size=(int(len(test) * te_frac), ))
    test = test[idxs, :, :]
print('Test size:\t', test.shape)

def get_dataloader(data, batch_size):
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

train_loader = get_dataloader(train, batch_size=cfg['model']['batch_size'])
test_loader = get_dataloader(test, batch_size=cfg['model']['batch_size'])

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD if cfg['model']['noise'] else BCE

epochs = cfg['model']['epochs']
device = cfg['model']['device']

model = MolecularVAE().to(device)
optimizer = optim.Adam(model.parameters(),
                       lr=cfg['model']['optimizer']['learning_rate'],
                       weight_decay=cfg['model']['optimizer']['weight_decay'])

def train_loop():
    model.train()
    train_loss = 0
    for batch_idx, data in tqdm(enumerate(train_loader)):
        data = data[0].transpose(1, 2).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data.transpose(1, 2), mu, logvar)
        loss.backward()
        train_loss += loss
        optimizer.step()
        if batch_idx % cfg['model']['verbose'] == 0:
            loss, current = loss.item(), batch_idx * len(data)
            print(f"loss: {loss / len(data):>7f}  [{current:>5d}/{len(train):>5d}]")
            idx = np.random.randint(len(data))
            pred = recon_batch.swapdims(1, 2).argmax(1)
            true = ''.join([idx2node[i] for i in data[idx, :, :].argmax(0).detach().cpu().numpy()])
            pred = ''.join([idx2node[i] for i in pred[idx, :].detach().cpu().numpy()])
            print(f'True:\t{true}')
            print(f'Pred:\t{pred}')
    return train_loss / len(train_loader.dataset)

def test_loop():
    model.eval()
    test_loss, correct = 0, 0
    results = {'recons': [], 'true': []}
    for batch_idx, data in tqdm(enumerate(test_loader)):
        data = data[0].transpose(1, 2).to(device)
        recon_batch, mu, logvar = model(data)
        results['recons'].append(recon_batch)
        results['true'].append(data)
        test_loss += loss_function(recon_batch, data.transpose(1, 2), mu, logvar).item()
        correct += (recon_batch.argmax(2) == torch.argmax(data.transpose(1, 2), dim=2)).type(torch.float).sum().item()
    test_loss /= len(test_loader) * len(data)
    correct /= len(test_loader.dataset) * seq_size
    # joblib.dump(results, f'./data/recons_{epoch}.pkl')
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss / len(test_loader.dataset)

for epoch in range(1, epochs + 1):
    train_loss = train_loop()
    test_loss = test_loop()

torch.save(model, 'ae_model.pth')

finish = datetime.now()
time_spent = finish - start
print(f'Total time spent: {time_spent}')