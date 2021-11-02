# molvaegraph
Molecular VAE with Graph Neural Network

data/preprocessing.py - SMILES to one-hot-encoded tensors
data/graph_preproc.py - SMILES to adjacency matricies and atom features joining
graph_encoder.py - sandbox for graph part of code
models.py - classic molecualar VAE
models_gcn.py - molVAE with joined graph encoder
train.py - classic molecualar VAE training
train_gcn.py - molVAE with joined graph encoder training
