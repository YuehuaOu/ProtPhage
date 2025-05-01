import csv
import random
import os
import ast
import math
import re
import gc
from re import L
import torch
import torch.nn as nn
import numpy as np
from transformers import AdamW
from tqdm import tqdm
from torch.optim import lr_scheduler
from model import *
from Get_embedding import *
from dataset import *
# from Get_Embedding import *
from Bio import SeqIO
from Bio.Seq import Seq
from transformers import T5EncoderModel, T5Tokenizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
# from distance_map import *
from config import config
import subprocess
import pickle




def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_pickle(target, path):
    f = open(path, 'wb')
    pickle.dump(target, f)
    f.close()


def load_pickle(path):
    f = open(path, 'rb')
    target = pickle.load(f)
    f.close()
    return target


def load_id_embedding(df, id):
    if isinstance(id, list):
        protein_embedding = np.array(df.loc[id]["protein_embedding"].apply(ast.literal_eval))
    else:
        protein_embedding = np.array(df.loc[[id]]["protein_embedding"].apply(ast.literal_eval))
    protein_embedding = np.vstack(protein_embedding)
    return protein_embedding


def evaluate(model, val_dataloader, embedding_df, criterion):
    model.eval()
    total_loss = 0
    epoch_labels, epoch_preds = [], []
    for ids, y in val_dataloader:
        x = take_embedding(embedding_df, ids)
        y = y.to(config.device)
        with torch.no_grad():
            outputs = model(x)
        loss = criterion(outputs.float(), y.float().squeeze())
        total_loss += loss.item()
        preds = outputs
        epoch_labels += list(y.cpu().numpy())
        epoch_preds += list(preds.argmax(1).cpu().numpy())
    epoch_f1 = f1_score(epoch_labels, epoch_preds)
    epoch_loss = total_loss / len(val_dataloader)
    return epoch_loss, epoch_f1



def test(model, test_dataloader, embedding_df):
    epoch_labels, epoch_preds = [], []
    model.eval()
    for ids, y in test_dataloader:
        x = take_embedding(embedding_df, ids)
        y = y.to(config.device)
        with torch.no_grad():
            outputs = model(x)
        preds = outputs
        epoch_labels += list(y.cpu().numpy())
        epoch_preds += list(preds.argmax(1).cpu().numpy())
    return epoch_labels, epoch_preds



def evaluate_muti(model, val_dataloader, embedding_df, criterion):
    model.eval()
    total_loss = 0
    epoch_labels, epoch_preds = [], []
    for ids, y in val_dataloader:
        x = take_embedding(embedding_df, ids)
        y = y.to(config.device)
        with torch.no_grad():
            outputs = model(x)
        loss = criterion(outputs.float(), y.float().squeeze())
        total_loss += loss.item()
        preds = outputs
        epoch_labels += list(y.cpu().numpy())
        epoch_preds += list(preds.argmax(1).cpu().numpy())
    epoch_f1 = f1_score(epoch_labels, epoch_preds, average='weighted')
    epoch_loss = total_loss / len(val_dataloader)
    return epoch_loss, epoch_f1


def test_muti(model, test_dataloader, embedding_df):
    epoch_labels, epoch_preds = [], []
    model.eval()
    for ids, y in test_dataloader:
        x = take_embedding(embedding_df, ids)
        y = y.to(config.device)
        with torch.no_grad():
            outputs = model(x)
        preds = outputs
        epoch_labels += list(y.cpu().numpy())
        epoch_preds += list(preds.argmax(1).cpu().numpy())
    return epoch_labels, epoch_preds

def test_muti_score(model, test_dataloader, embedding_df):
    epoch_labels, epoch_preds, epoch_scores = [], [], []
    model.eval()
    for ids, y in test_dataloader:
        x = take_embedding(embedding_df, ids)
        y = y.to(config.device)
        with torch.no_grad():
            outputs = model(x)
        preds = outputs
        max_scores, predicted_classes = preds.max(dim=1)
        epoch_labels += list(y.cpu().numpy())
        epoch_preds += list(preds.argmax(1).cpu().numpy())
        epoch_scores += list(max_scores.cpu().numpy())
    return epoch_labels, epoch_preds, epoch_scores