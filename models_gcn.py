import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

with open('conf.yml') as f:
    cfg = load(f.read(), Loader=Loader)

INPUT_SIZE = cfg['data']['alphabet_size']
SEQ_SIZE = cfg['data']['seq_size']
GRAPH_SIZE = cfg['data']['graph_size']
ATOMIC_FEATS_SIZE = cfg['data']['atomic_feats_size']
GRAPH_HIDDEN_SIZE = cfg['model']['graph']['output_size']
GRAPH_OUTPUT_SIZE = cfg['model']['graph']['hidden_size']
VIRT_NODE = cfg['data']['virtual_node_idx'] >= 0

class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, graph_size, device):
        super(GCNConv, self).__init__()
        self.graph_size = graph_size
        self.W = nn.Parameter(torch.rand(in_channels, out_channels, requires_grad=True))
        self.device = device

    def forward(self, X):
        A = X[:, :, :self.graph_size]
        X = X[:, :, self.graph_size:]
        batch_size = A.shape[0]
        E = torch.eye(self.graph_size).expand(batch_size, self.graph_size, self.graph_size).to(self.device)
        A_hat = (A + E).double()
        D = torch.diag_embed(torch.sum(A, 2), dim1=-2, dim2=-1).double()
        D = torch.linalg.pinv(D).sqrt()
        A_hat = torch.matmul(torch.matmul(D, A_hat), D)
        output = torch.relu(torch.matmul(torch.matmul(A_hat, X.double()), self.W.double()))
        output = torch.cat([A, output], dim=2)
        return output


class MolecularGraphVAE(nn.Module):
    def __init__(self, device, graph_size=GRAPH_SIZE):
        super(MolecularGraphVAE, self).__init__()

        # The input filter dim should be 35
        #  corresponds to the size of CHARSET
        self.graph_size = graph_size
        if not VIRT_NODE:
            self.graph_size -= 1
        self.conv1d1 = nn.Conv1d(INPUT_SIZE, 9, kernel_size=9)
        self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv1d3 = nn.Conv1d(9, 20, kernel_size=11)
        self.fc0 = nn.Linear(660, 435)
        self.fc11 = nn.Linear(435 + GRAPH_OUTPUT_SIZE, 292)
        self.fc12 = nn.Linear(435 + GRAPH_OUTPUT_SIZE, 292)
        self.gcnconv1 = GCNConv(ATOMIC_FEATS_SIZE, GRAPH_HIDDEN_SIZE, self.graph_size, device)
        self.gcnconv2 = GCNConv(GRAPH_HIDDEN_SIZE, GRAPH_OUTPUT_SIZE, self.graph_size, device)

        self.fc2 = nn.Linear(292, 292)
        self.gru = nn.GRU(292, 501, 3, batch_first=True)
        self.fc3 = nn.Linear(501, INPUT_SIZE)

    def encode_graph(self, graph_x):
        if VIRT_NODE:
            h = self.gcnconv1(graph_x)
            z = self.gcnconv2(h)
            return z[:, cfg['data']['virtual_node_idx'], self.graph_size:]
        else:
            graph_x = graph_x[:, 1:, 1:]
            h = self.gcnconv1(graph_x)
            z = self.gcnconv2(h)
            return z[:, :, self.graph_size:].sum(dim=1)

    def encode(self, seq_x, graph_x):
        h = F.relu(self.conv1d1(seq_x))
        h = F.relu(self.conv1d2(h))
        h = F.relu(self.conv1d3(h))
        h = h.view(h.size(0), -1)
        h = F.selu(self.fc0(h))
        h = torch.cat([h, graph_x], dim=1).float()
        return self.fc11(h), self.fc12(h)

    def reparametrize(self, mu, logvar):
        if self.training:
            if cfg['model']['noise']:
                std = torch.exp(0.5 * logvar)
                eps = 1e-2 * torch.randn_like(std)
                w = eps.mul(std).add_(mu)
                return w
            else:
                return mu
        else:
            return mu

    def decode(self, z):
        z = F.selu(self.fc2(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, SEQ_SIZE, 1)
        out, h = self.gru(z)
        out_reshape = out.contiguous().view(-1, out.size(-1))
        y0 = F.softmax(self.fc3(out_reshape), dim=1)
        y = y0.contiguous().view(out.size(0), -1, y0.size(-1))
        return y

    def forward(self, seq_x, graph_x):
        graph_x = self.encode_graph(graph_x)
        mu, logvar = self.encode(seq_x, graph_x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar