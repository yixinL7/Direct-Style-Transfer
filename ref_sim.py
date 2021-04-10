import torch
from sim_models import WordAveraging, GateModel
from sim_utils import Example
from nltk.tokenize import TreebankWordTokenizer
import sentencepiece as spm

tok = TreebankWordTokenizer()

def make_example(sentence, model):
    sentence = sentence.lower()
    sentence = " ".join(tok.tokenize(sentence))
    sentence = sp.EncodeAsPieces(sentence)
    wp1 = Example(" ".join(sentence))
    wp1.populate_embeddings(model.vocab)
    return wp1

model = torch.load('result/sim.pt',
                   map_location='cpu')
state_dict = model['state_dict']
vocab_words = model['vocab_words']
args = model['args']
# turn off gpu
args.gpu = -1
model = WordAveraging(args, vocab_words)
model.load_state_dict(state_dict, strict=True)

sp = spm.SentencePieceProcessor()
sp.Load('sim/sim.sp.30k.model')

def sim_score_new(refs, hyps):
    scores = 0
    for i in range(len(refs)):
        s1 = make_example(refs[i], model)
        s2 = make_example(hyps[i], model)
        wx1, wl1, wm1 = model.torchify_batch([s1])
        wx2, wl2, wm2 = model.torchify_batch([s2])
        scores += model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2).item()
    return scores / float(len(refs))