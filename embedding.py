import torch
import re
import gc
from tqdm import tqdm
import numpy as np
from Bio import SeqIO
from transformers import T5EncoderModel, T5Tokenizer





def get_embedding(sequences_Example):
    # Automatic extracted features
    tokenizer = T5Tokenizer.from_pretrained("/root/nas-private/ProtT5_model",  do_lower_case=False)
    model = T5EncoderModel.from_pretrained("/root/nas-private/ProtT5_model")
    gc.collect()
    model = model.eval()
    features = []
    flag = 0
    for i in tqdm(range(len(sequences_Example))):
        sequences_Example_i = sequences_Example[i]
        if len(sequences_Example_i) <= 6000 and flag == 0:
            device = torch.device('cuda:0')
            model = model.to(device)
            flag = 1
        elif len(sequences_Example_i) > 6000 and flag == 1:
            device = torch.device('cpu')
            model = model.to(device)
            flag = 0
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        features_normalize[i] = np.sum(features[i], axis=0) / len(features[i])
    return features_normalize
