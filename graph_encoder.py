import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm
from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

with open('conf.yml') as f:
    cfg = load(f.read(), Loader=Loader)

GRAPH_SIZE = cfg['data']['graph_size']
ATOMIC_FEATS_SIZE = cfg['data']['atomic_feats_size']
GRAPH_HIDDEN_SIZE = cfg['model']['graph']['output_size']
GRAPH_OUTPUT_SIZE = cfg['model']['graph']['hidden_size']


class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.W = nn.Parameter(torch.rand(in_channels, out_channels, requires_grad=True))

    def forward(self, X):
        A = X[:, :, :GRAPH_SIZE]
        X = X[:, :, GRAPH_SIZE:]
        batch_size = A.shape[0]
        A_hat = (A + torch.eye(GRAPH_SIZE).expand(batch_size, GRAPH_SIZE, GRAPH_SIZE)).double()
        D = torch.diag_embed(torch.sum(A, 2), dim1=-2, dim2=-1).double()
        D = torch.linalg.pinv(D).sqrt()
        A_hat = torch.matmul(torch.matmul(D, A_hat), D)
        output = torch.relu(torch.matmul(torch.matmul(A_hat, X.double()), self.W.double()))
        output = torch.cat([A, output], dim=2)
        return output


class GCNEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)

    def forward(self, X):
        H = self.conv1(X)
        Z = self.conv2(H)
        return Z[:, [0], GRAPH_SIZE:]

# A = torch.randint(0, 2, size=(2, 3, 3))
# print(A)
# A_hat = (A + torch.eye(3).expand(2, 3, 3)).double()
# print(A_hat)
# D = torch.diag_embed(torch.sum(A, 2), dim1=-2, dim2=-1).double()#.inverse().sqrt()
# print(D)
# D = torch.linalg.pinv(D).sqrt()
# print(D)
# A_hat = torch.matmul(torch.matmul(D, A_hat), D)
# print(A_hat)

with np.load('./data/graph_tensors.npz') as loader:
    # A_train = loader['arr_0']
    # X_train = loader['arr_1']
    X_test = torch.cat(
        [torch.Tensor(loader['arr_2']), torch.Tensor(loader['arr_3'])], dim=2)

# print(X_test[:, :, :GRAPH_SIZE].shape)

def get_dataloader(data, batch_size):
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

test_loader = get_dataloader(X_test, batch_size=cfg['model']['batch_size'])
graph_model = GCNEncoder(ATOMIC_FEATS_SIZE, GRAPH_HIDDEN_SIZE, GRAPH_OUTPUT_SIZE)
graph_layer = GCNConv(ATOMIC_FEATS_SIZE, GRAPH_OUTPUT_SIZE)

for batch_idx, data in tqdm(enumerate(test_loader)):
    data_t = graph_model(data[0])
    break

print(data[0].shape)
print(data_t.shape)

