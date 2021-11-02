# molvaegraph
Molecular VAE with Graph Neural Network

- data/preprocessing.py - SMILES to one-hot-encoded tensors
- data/graph_preproc.py - SMILES to adjacency matricies and atom features joining
- graph_encoder.py - sandbox for graph part of code


- models.py - vanila molecualar VAE
- train.py - vanila molecualar VAE training
- reconstruction.py - reconstruction script with vanila molecular VAE
- ae_sample.csv - reconstruction results


- models_gcn.py - molVAE with joined graph encoder
- train_gcn.py - molVAE with joined graph encoder training

