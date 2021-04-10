import torch
from nltk.translate.bleu_score import sentence_bleu
import os

VERBOSE = True
class MyPrint():
    def __init__(self, f, tofile=True):
        self.tofile = tofile
        if tofile:
            self.f = open(f, "w")

    def myprint(self, s=None):
        if VERBOSE:
            if s is None:
                print()
                if self.tofile:
                    self.f.write("\n")
            else:
                print(s)
                if self.tofile:
                    if isinstance(s, str):
                        self.f.write(s)
                        self.f.write("\n")
                    else:
                        self.f.write(str(s))
                        self.f.write("\n")

BACKEND = None

DEFALUT = "./log/default.txt"

PRINT = False
def start_log(f, tofile=True):
    global BACKEND
    BACKEND = MyPrint(f, tofile)
    #BACKEND.myprint("hello")

def myprint(s=None):
    global BACKEND
    if BACKEND is None:
        BACKEND = MyPrint(DEFALUT)
    BACKEND.myprint(s)

def translate(vocab, idx):
    text = []
    eos = vocab.eos()
    for x in idx:
        if x == eos:
            break
        text.append(vocab.i2w(x))
    return " ".join(text)

def translate_(vocab, idx):
    text = []
    eos = vocab.eos()
    for x in idx:
        if x == eos:
            break
        text.append(vocab.i2w(x))
    return text

def decode(vocab, idx, sp):
    text = translate_(vocab, idx)
    text = sp.DecodePieces(text)
    return text.split(" ")

def load_setting(fdir):
    with open(fdir, "r") as f:
        config = yaml.load(f)
    #print(config)
    return config

def isnan(x):
    mask = torch.isnan(x)
    zeros = torch.zeros_like(x, dtype=torch.uint8)
    return not torch.equal(mask, zeros)

def clean(x, eos=3):
    r = []
    for x_i in x:
        tmp = [0] * len(x_i)
        for i, w in enumerate(x_i):
            tmp[i] = w
            if w == eos:
                break
        r.append(torch.tensor(tmp))
    return torch.stack(r, dim=0)

def found(x, s):
    for x_i in x:
        if x_i[:len(s)] == s:
            return x_i




