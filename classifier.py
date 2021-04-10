import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import Dataloader
import pickle
import numpy as np
from utils import isnan
import json

KERNEL = [(1, 256), (2, 256), (3, 256), (4, 256), (5, 256)]
AMAZON_KERNEL = [(3, 1024), (4, 1024), (5, 1024)]
# KERNEL = [(1, 256), (2, 256), (3, 256), (4, 256), (5, 256)]
BATCH_SIZE = 64
IS_CUDA = True
EPOCH = 20


class AdvDisNet(nn.Module):
    """ Discriminator with CNN and Highway  """
    def __init__(self, word_num=9366, kernels=KERNEL, word_dim=512, hidden_dim=512, output_dim=1, drop_out=0.1, activation="sigmoid"):
        """
        kernels : cnn kernels, list of tuple (width, num)
        word_dim : word vector size
        output_dim : output size
        """
        super(AdvDisNet, self).__init__()
        self.embed = nn.Embedding(word_num, word_dim)
        self.cnns = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=t[1], kernel_size=(t[0], word_dim), stride=1) for t in kernels])
        nums = 0
        for t in kernels:
            nums += t[1]
        self.drop = nn.Dropout(p=drop_out)
        self.linear = nn.Linear(nums, hidden_dim)
        self.out = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.style = nn.Embedding(2, hidden_dim)
        if activation == "hardtanh":
            self.activation = nn.Hardtanh()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "softsign":
            self.activation = nn.Softsign()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softplus":
            self.activation = nn.Softplus()
        else:
            print("not implemented!")
            quit()

    def feature(self, x, mode="regular"):
        r = []
        if mode == "regular":
            x = self.embed(x)
        elif mode == "approximate":
            x = torch.matmul(x, self.embed.weight)
        else:
            x = self.embed(x)
            x = x + torch.randn_like(x) * 0.1
        x = torch.unsqueeze(x, dim=1)
        for cnn in self.cnns:
            y = F.relu(torch.squeeze(cnn(x), dim=3))
            y = F.max_pool1d(y, y.size(2)).squeeze(2)
            r.append(y)
        r = torch.cat(r, dim=1)
        r = self.drop(r)
        return self.linear(r)

    def forward(self, x, style):
        p = self.feature(x)
        p = torch.cat((p, self.style(style)), dim=1)
        p = self.activation(self.out(p))
        return p

    def approximate(self, x, style):
        p = self.feature(x, "approximate")
        p = torch.cat((p, self.style(style)), dim=1)
        p = self.activation(self.out(p))
        return p

    def noise(self, x, style):
        p = self.feature(x, "noise")
        p = self.activation(self.out(p))
        return p
    

class RNNDisNet(nn.Module):
    def __init__(self, word_num, word_dim=512, hidden_size=512, num_layers=2, dropout=0.1):
        super(RNNDisNet, self).__init__()
        self.embed = nn.Embedding(word_num, word_dim, 0)
        self.lstm = nn.LSTM(input_size=word_dim, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        self.layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        _, (t, _) = self.lstm(self.embed(x))
        t = t.view(self.layers, 2, -1, self.hidden_size)[-1]
        t = torch.transpose(t, 0, 1).contiguous()
        t = t.view(-1, 2 * self.hidden_size)
        p = F.sigmoid(self.mlp(t))
        return p

    def approximate(self, x):
        embedding = torch.matmul(x, self.embed.weight)
        _, (t, _) = self.lstm(embedding)
        t = t.view(self.layers, 2, -1, self.hidden_size)[-1]
        t = torch.transpose(t, 0, 1).contiguous()
        t = t.view(-1, 2 * self.hidden_size)
        p = F.sigmoid(self.mlp(t))
        return p

    def score(self, x):
        _, (t, _) = self.lstm(self.embed(x))
        t = t.view(self.layers, 2, -1, self.hidden_size)[-1]
        t = torch.transpose(t, 0, 1).contiguous()
        t = t.view(-1, 2 * self.hidden_size)
        t = self.mlp(t)
        return t

def classifer_test(model, tokenizer, dataloader, BATCH_SIZE):
    model = model.eval()
    all_cnt = 0
    correct_cnt = 0
    with torch.no_grad():
        for (j, batch) in enumerate(dataloader):
            true_probs = model(batch["src_text"], batch["style"])
            false_probs = model(batch["src_text"], 1 - batch["style"])
            probs = true_probs.cpu().data.numpy()
            probs = np.concatenate(probs)
            target = batch["style"].cpu().data.numpy()
            all_cnt += BATCH_SIZE
            correct_cnt += np.sum(probs > 0.5)
            probs = false_probs.cpu().data.numpy()
            probs = np.concatenate(probs)
            target = batch["style"].cpu().data.numpy()
            all_cnt += BATCH_SIZE
            correct_cnt += np.sum(probs < 0.5)
    model.train()
    return correct_cnt / all_cnt