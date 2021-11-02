"""
SMILES STRINGS PREPROCESSNG
"""
import os
import random

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


SEED = 19
random.seed(SEED)
np.random.seed(SEED)
PROJ_PATH = os.getcwd()
INPUT_DATA_PATH = os.path.join(PROJ_PATH, 'data', 'raw')
OUTPUT_DATA_PATH = os.path.join(PROJ_PATH, 'data', 'preprocessed')

BLACK_SET = {'B', 'r', 'l'}
ADD_SET = {'Br', 'Cl'}
SOM = '>'
EOM = '<'
PAD = '_'

train = pd.read_csv(os.path.join(INPUT_DATA_PATH, 'train.csv'))['SMILES']
test = pd.read_csv(os.path.join(INPUT_DATA_PATH, 'test.csv'))['SMILES']
test_sc = pd.read_csv(os.path.join(INPUT_DATA_PATH, 'test_scaffolds.csv'))['SMILES']

seq_len = max(train.apply(len).max(), test.apply(len).max(), test_sc.apply(len).max())
print(f'Max SMILES length is {seq_len}')

train = train.apply(lambda seq: SOM + seq + EOM + PAD * (seq_len - len(seq)))
test = test.apply(lambda seq: SOM + seq + EOM + PAD * (seq_len - len(seq)))
test_sc = test_sc.apply(lambda seq: SOM + seq + EOM + PAD * (seq_len - len(seq)))


def get_alphabet(data, black_set, add_set):
    string = ''.join(data)
    alphabet = set(string)
    alphabet -= black_set
    for item in add_set:
        alphabet.add(item)
    alphabet = list(alphabet)
    return {i: alphabet[i] for i in range(len(alphabet))}


idx2node = get_alphabet(train, BLACK_SET, ADD_SET)
node2idx = {node: idx for idx, node in idx2node.items()}
print(node2idx)

def get_ohe_tensor(data):
    data_ohe = np.zeros((len(data), seq_len + 2, len(node2idx)))
    for i, seq in tqdm(enumerate(data), total=len(data)):
        j = 0
        while j < len(seq):
            try:
                node = seq[j:j+2]
                k = node2idx[node]
            except KeyError:
                node = seq[j]
                k = node2idx[node]
            data_ohe[i, j, k] = 1
            j += len(node)
    return data_ohe

train_ohe = get_ohe_tensor(train)
test_ohe = get_ohe_tensor(test)
test_sc_ohe = get_ohe_tensor(test_sc)

joblib.dump(node2idx, os.path.join(OUTPUT_DATA_PATH, 'node2idx.pkl'))
joblib.dump(idx2node, os.path.join(OUTPUT_DATA_PATH, 'idx2node.pkl'))
joblib.dump(train_ohe, os.path.join(OUTPUT_DATA_PATH, 'train.pkl'))
joblib.dump(test_ohe, os.path.join(OUTPUT_DATA_PATH, 'test.pkl'))
joblib.dump(test_sc_ohe, os.path.join(OUTPUT_DATA_PATH, 'test_sc.pkl'))
