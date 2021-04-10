import torch
import random
import pickle
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import os
import time

def pad(x):
    return pad_sequence(x, batch_first=True, padding_value=0)

def mypad(X, pad=0):
    """
    zero-pad sequnces to same length then stack them together
    """  
    maxlen = max([x.size(0) for x in X])
    Y = []
    for x in X:
        padlen = maxlen - x.size(0)
        if padlen > 0:
            paddings = torch.ones(padlen, requires_grad=True).type(x.type()) * pad
            x_ = torch.cat((x, paddings), 0)
            Y.append(x_)
        else:
            Y.append(x)
    return torch.stack(Y)


class GPTLoader(object):
    """
    bert loader
    """
    def __init__(self, data, tokenizer, batch_size, use_gpu=False, shuffle=False, input_maxlen=None):
        """
        data: list of json {"sent", "style"}
        """
        self.data = data
        self.batch_size = batch_size
        self.num = len(self.data)
        self.count = 0
        self.iters = int(self.num / batch_size)
        self.use_gpu = use_gpu
        self.shuffle = shuffle
        self.input_maxlen = input_maxlen
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id
        self.style_token = [tokenizer.vocab_size, tokenizer.vocab_size + 1]
        # print(self.style_token)
        # print(self.tokenizer)
        # print(self.pad)
        # print(self.bos_id)
        # print(self.eos_id)
        # self.style_token = 
        if self.shuffle:
            random.shuffle(self.data)
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.count == self.iters:
            self.count = 0
            if self.shuffle:
                random.shuffle(self.data)
            raise StopIteration()
        else:
            batch = self.data[self.count * self.batch_size : (self.count + 1) * self.batch_size]
            text = [self.tokenizer.encode(x["sent"], add_special_tokens=False, add_prefix_space=True) for x in batch]
            # sample = [self.tokenizer.encode(x["sent"] + self.style_token[x["style"]] + x["sent"], add_special_tokens=False, add_prefix_space=True) for x in batch]
            idxs = [(i, len(text[i])) for i in range(self.batch_size)]
            idxs = sorted(idxs, key=lambda x : x[1], reverse=True)
            self.count += 1
            if self.input_maxlen is not None:
                src_text = mypad([torch.LongTensor([self.bos_id] + text[i][ : self.input_maxlen] + [self.eos_id]) for (i, _) in idxs])
                # tgt_text = mypad([torch.LongTensor([self.bos_id] + text[i][: self.input_maxlen]) for (i, _) in idxs])
                length = torch.LongTensor([min(i, self.input_maxlen) + 2 for (_, i) in idxs])
            else:
                src_text = mypad([torch.LongTensor([self.bos_id] + text[i] + [self.eos_id]) for (i, _) in idxs])
                # tgt_text = mypad([torch.LongTensor([self.bos_id] + text[i]) for (i, _) in idxs])
                length = torch.LongTensor([i + 2 for (_, i) in idxs])
            tokens = [batch[i]["sent"] for (i, _) in idxs]
            style = torch.LongTensor([batch[i]["style"] for (i, _) in idxs])
            style_tokens = torch.LongTensor([self.style_token[batch[i]["style"]] for (i, _) in idxs])
            transfer_tokens = torch.LongTensor([self.style_token[1 - batch[i]["style"]] for (i, _) in idxs])
            if self.use_gpu:
                src_text = src_text.cuda()
                length = length.cuda()
                style = style.cuda()
                style_tokens = style_tokens.cuda()
                transfer_tokens = transfer_tokens.cuda()
            return {"src_text": src_text, "length": length, "style": style, "tokens": tokens, "style_tokens": style_tokens, "transfer_tokens": transfer_tokens, "original_index": [i for (i, _) in idxs]}

    def get(self):
        try:
            data = self.__next__()
        except StopIteration:
            data = self.__next__()
        return data

class GPTRefLoader(object):
    """
    bert loader
    """
    def __init__(self, data, tokenizer, batch_size, use_gpu=False, shuffle=False, input_maxlen=None):
        """
        data: list of json {"sent", "style"}
        """
        # ref = [x.strip().split(" ") for x in ref]
        # for (i, x) in enumerate(ref):
        #     data[i]["ref"] = x
        self.data = data
        self.batch_size = batch_size
        self.num = len(self.data)
        self.count = 0
        self.iters = int(self.num / batch_size)
        self.use_gpu = use_gpu
        self.shuffle = shuffle
        self.input_maxlen = input_maxlen
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id
        self.style_token = [tokenizer.vocab_size, tokenizer.vocab_size + 1]
        # self.style_token = 
        if self.shuffle:
            random.shuffle(self.data)
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.count == self.iters:
            self.count = 0
            if self.shuffle:
                random.shuffle(self.data)
            raise StopIteration()
        else:
            batch = self.data[self.count * self.batch_size : (self.count + 1) * self.batch_size]
            text = [self.tokenizer.encode(x["sent"], add_special_tokens=False, add_prefix_space=True) for x in batch]
            # sample = [self.tokenizer.encode(x["sent"] + self.style_token[x["style"]] + x["sent"], add_special_tokens=False, add_prefix_space=True) for x in batch]
            idxs = [(i, len(text[i])) for i in range(self.batch_size)]
            idxs = sorted(idxs, key=lambda x : x[1], reverse=True)
            self.count += 1
            if self.input_maxlen is not None:
                src_text = mypad([torch.LongTensor([self.bos_id] + text[i][ : self.input_maxlen] + [self.eos_id]) for (i, _) in idxs])
                # tgt_text = mypad([torch.LongTensor([self.bos_id] + text[i][: self.input_maxlen]) for (i, _) in idxs])
                length = torch.LongTensor([min(i, self.input_maxlen) + 2 for (_, i) in idxs])
            else:
                src_text = mypad([torch.LongTensor([self.bos_id] + text[i] + [self.eos_id]) for (i, _) in idxs])
                # tgt_text = mypad([torch.LongTensor([self.bos_id] + text[i]) for (i, _) in idxs])
                length = torch.LongTensor([i + 2 for (_, i) in idxs])
            tokens = [batch[i]["sent"] for (i, _) in idxs]
            style = torch.LongTensor([batch[i]["style"] for (i, _) in idxs])
            style_tokens = torch.LongTensor([self.style_token[batch[i]["style"]] for (i, _) in idxs])
            transfer_tokens = torch.LongTensor([self.style_token[1 - batch[i]["style"]] for (i, _) in idxs])
            ref_tokens = mypad([torch.LongTensor([self.bos_id] + self.tokenizer.encode(batch[i]["ref"], add_special_tokens=False, add_prefix_space=True) + [self.eos_id]) for (i, _) in idxs])
            if self.use_gpu:
                src_text = src_text.cuda()
                length = length.cuda()
                style = style.cuda()
                style_tokens = style_tokens.cuda()
                transfer_tokens = transfer_tokens.cuda()
                ref_tokens = ref_tokens.cuda()
            ref = [batch[i]["ref"] for (i, _) in idxs]
            return {"src_text": src_text, "length": length, "style": style, "tokens": tokens, "ref": ref, "style_tokens": style_tokens, "transfer_tokens": transfer_tokens, "ref_tokens": ref_tokens}

    def get(self):
        try:
            data = self.__next__()
        except StopIteration:
            data = self.__next__()
        return data