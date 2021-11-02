import joblib
import numpy
import numpy as np
from ogb.utils.features import atom_to_feature_vector
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

GRAPH_SIZE = 27 + 1
N_FEATURES = 9

def smiles_parsing(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    A = Chem.GetAdjacencyMatrix(mol)
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    X = np.array(atom_features_list, dtype=np.int64)
    return A, X


def add_virtual_node(A, X):
    A = np.concatenate([np.ones((A.shape[0], 1)).astype(int), A], axis=1)
    A = np.concatenate([np.ones((1, A.shape[1])).astype(int), A], axis=0)
    A[0, 0] = 0
    X = np.concatenate([-np.ones((1, 9)).astype(int), X], axis=0)
    return A, X


def get_graph_data(data_path):
    max_size = 0
    smiles_arr = pd.read_csv(data_path)['SMILES']
    A_tensor = np.zeros((len(smiles_arr), GRAPH_SIZE, GRAPH_SIZE))
    X_tensor = np.zeros((len(smiles_arr), GRAPH_SIZE, N_FEATURES))
    for i, smiles in tqdm(enumerate(smiles_arr), total=len(smiles_arr)):
        A, X = smiles_parsing(smiles)
        if A.shape[0] > max_size:
            max_size = A.shape[0]
            print(max_size)
        A, X = add_virtual_node(A, X)
        A_tensor[i, :A.shape[0], :A.shape[0]] = A
        X_tensor[i, :X.shape[0], :N_FEATURES] = X
    return A_tensor.astype(int), X_tensor.astype(int)


A_train, X_train = get_graph_data('./raw/train.csv')
A_test, X_test = get_graph_data('./raw/test.csv')
# A_test_sc, X_test_sc = get_graph_data('./raw/test_scaffolds.csv')

# result = (A_train, X_train), (A_test, X_test)
# joblib.dump(result, 'graph_tensors.pkl')

np.savez_compressed('graph_tensors.npz', A_train, X_train, A_test, X_test)