from torch.utils.data import Dataset
from utils import *
import utils
from transformers import T5EncoderModel, T5Tokenizer
from config import config
import numpy as np
import torch



class MyDataset(Dataset):
    def __init__(self, seq_ids, labels):
        self.seq_ids = seq_ids
        self.labels = labels
        Y = self.load_data()
        self.y = Y


    def __len__(self):
        return len(self.seq_ids)


    def __getitem__(self, index):
        return self.seq_ids[index], self.y[index]

    def load_data(self):
        Y = torch.from_numpy(np.array(self.labels)).long()
        return Y