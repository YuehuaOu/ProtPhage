from Get_embedding import *
from config import config
from Bio import SeqIO
import pandas as pd

# df = pd.read_csv(config.case_study_data)
# proteins, ids = [], []

# for index, row in df.iterrows():
#     proteins.append(row['sequence'])
#     ids.append(row['accession'])


proteins, ids = [], []
for rec in SeqIO.parse(config.data, "fasta"):
    id = str(rec.id)
    seq = str(rec.seq)
    proteins.append(seq)
    ids.append(id)
    
compute_embedding(proteins, ids, config.embedding_file)