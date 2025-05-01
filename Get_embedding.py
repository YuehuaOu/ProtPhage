import pandas as pd
import ast
import numpy as np
import torch
from config import config
import numpy as np
from embedding import get_embedding



def compute_embedding(proteins, ids, embedding_file):
    seq = []
    for i in range(len(proteins)):
        zj = ''
        for j in range(len(proteins[i]) - 1):
            zj += proteins[i][j] + ' '
        zj += proteins[i][-1]
        seq.append(zj)
    emb = get_embedding(seq)
    data = {'id': ids, 'protein_embedding': emb.tolist()}
    df = pd.DataFrame(data)
    df.to_excel(embedding_file, index=False)


def take_embedding(dataframe, ids):
    # df = pd.read_excel(embeddiing_file)
    # df.set_index("id", inplace=True)
    df = dataframe
    ids = list(ids)
    protein_embedding = np.array(df.loc[ids]["protein_embedding"].apply(ast.literal_eval))
    protein_embedding = np.vstack(protein_embedding)
    protein_embedding = torch.from_numpy(protein_embedding).float().to(config.device)
    return protein_embedding