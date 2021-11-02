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
DATA_PATH = './data/preprocessed'
with open('conf.yml') as f:
    cfg = load(f.read(), Loader=Loader)

device = cfg['model']['device']
model = torch.load('ae_model.pth').to(device)
model.eval()

alphabet_size = cfg['data']['alphabet_size']
seq_size = cfg['data']['seq_size']
node2idx = joblib.load(os.path.join(DATA_PATH, 'node2idx.pkl'))
idx2node = joblib.load(os.path.join(DATA_PATH, 'idx2node.pkl'))
test = torch.Tensor(joblib.load(os.path.join(DATA_PATH, 'test.pkl')))

def get_dataloader(data, batch_size):
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

test_loader = get_dataloader(test, batch_size=cfg['generation']['batch_size'])

true_list, pred_list = [], []

for batch_idx, data in tqdm(enumerate(test_loader)):
    data_batch = data[0].transpose(1, 2).to(device)
    recon_batch, mu, logvar = model(data_batch)
    pred_batch = recon_batch.swapdims(1, 2).argmax(1)
    for idx in range(cfg['generation']['batch_size']):
        true = ''.join([idx2node[i] for i in data_batch[idx, :, :].argmax(0).detach().cpu().numpy()])
        pred = ''.join([idx2node[i] for i in pred_batch[idx, :].detach().cpu().numpy()])
        true_list.append(true)
        pred_list.append(pred)
    break

result = pd.DataFrame(index=range(cfg['generation']['batch_size']),
                      columns=['smiles', 'reconstructed'])
result['smiles'] = true_list
result['reconstructed'] = pred_list
result.to_csv('ae_sample.csv', sep='\t', index=False)