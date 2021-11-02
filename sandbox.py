import numpy as np
import torch

from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

with open('conf.yml') as f:
    cfg = load(f.read(), Loader=Loader)

with np.load('./data/graph_tensors.npz') as loader:
    # print(loader['arr_2'].mean())
    test_graph = torch.Tensor(loader['arr_3'])
    # test_graph = (test_graph - test_graph.min()) / (test_graph.max())

print(
    test_graph.max(dim=1)[0].max(dim=0)[0],
    test_graph.min(dim=1)[0].min(dim=0)[0]
)#, test_graph.min(), test_graph.mean())

max_tensor = test_graph.max(dim=1)[0].max(dim=0)[0].unsqueeze(0).unsqueeze(0)
max_tensor = max_tensor.expand(*test_graph.shape)
print(torch.nan_to_num(test_graph / max_tensor)[:2, :, :])