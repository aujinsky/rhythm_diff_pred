
import math
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import re
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from torch.nn import LSTM, Softmax
from torch.nn.utils.rnn import pad_sequence
from einops import repeat, rearrange
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from tqdm import tqdm
from utils import dataset
import numpy as np
import utils.calculate as cal

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000, device=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

def gen_src_mask(total_len, len_list, device=0):
    batch_len = len(len_list)
    zero = torch.zeros(batch_len, total_len, device=device)
    for tens, t in zip(zero, len_list):
        mask = torch.ones(total_len-t, device=device) 
        tens[t:] = mask
    ret = zero.bool()
    return ret

# Encoder-only model that pretrains via osu mania pp scores. 

class RhythmRAE(nn.Module):
    def __init__(self, num_keys = 4, d_model = 16, nhead = 8, seq_len = 5000, nlayers = 4, mask= True):
        super(RhythmRAE, self).__init__()
        self.num_keys = num_keys
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.seq_len = seq_len
        self.mask = mask
        self.hidden = self.d_model

        self.src_emb = nn.Linear(self.num_keys, self.hidden)
        encoder_layer = nn.TransformerEncoderLayer(self.hidden, nhead, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pos_encoding = PositionalEncoding(self.hidden, max_len=seq_len)
        self.linear1 = nn.Linear(self.hidden, 4)
        self.linear2 = nn.Linear(4, 1)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim = 2)


    def forward(self, src, src_padding_mask = None): 
        #src = self.pos_encoding(src)
        src = self.src_emb(src)
        h = self.encoder(src, src_key_padding_mask = src_padding_mask)
        h = self.linear1(h)
        h = (torch.transpose(src_padding_mask, 0, 1).unsqueeze(2) * h).sum(dim = 0) / (~src_padding_mask).sum(dim = 1).unsqueeze(1) # this whole process indorses non - note timestamp obsolete, which can be concerning for further use
        out = self.linear2(h)
        return out




if __name__=="__main__":
    random_seed = 42
    torch.manual_seed(random_seed)

    dataset_osu = dataset.OsuManiaDataset(
        fpath = 'osumania_data\mapping_list.csv'
    )
    debug_data = dataset_osu.collate_fn([dataset_osu[0], dataset_osu[5]])
    mask = gen_src_mask(debug_data[1][1], debug_data[1][0])
    debug_model = RhythmRAE().to(0)
    dummy_result = debug_model(debug_data[0], mask)
    import IPython; IPython.embed(); exit(1)