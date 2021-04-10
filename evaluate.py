import torch
import torch.nn as nn
import Dataloader
import argparse
import nltk
import gpt_utils
from nltk.tokenize import TreebankWordTokenizer
from ref_sim import sim_score_new
import json
import subprocess as sp
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tok = TreebankWordTokenizer()

def s_bleu(ref, hyp, f):
    num = len(ref)
    score = 0
    for i in range(num):
        score += nltk.translate.bleu_score.sentence_bleu(ref[i], hyp[i], smoothing_function=f)
    return score / float(num)


def clean(idxs, tokenizer):
    r = []
    eos_id = tokenizer.eos_token_id
    for x in idxs:
        if x == eos_id:
            break
        else:
            r.append(x)
    return r


def get_len(idxs, tokenizer):
    i = 0
    eos_id = tokenizer.eos_token_id
    for x in idxs:
        i += 1
        if x == eos_id:
            return i
    return i


def generate_output(generator, args, dataloader, tokenizer, BATCH_SIZE=1, fname=None, generate=False, dname="yelp", pos_num=None):
    # start experiment
    with torch.no_grad():
        batch_size = BATCH_SIZE
        method = nltk.translate.bleu_score.SmoothingFunction(0.000000001).method1
        # load data 
        # build model
        hyp = []
        for (j, batch) in enumerate(dataloader):
            transfer_text = torch.cat((batch["src_text"], batch["transfer_tokens"].unsqueeze(1)), dim=1)
            cur_len = transfer_text.size(1)
            _, probs = gpt_utils.generate(generator, transfer_text, cur_len=cur_len, max_length=int(cur_len * 2 - 1), pad_token_id=tokenizer.pad_token_id,
             eos_token_ids=tokenizer.eos_token_id, batch_size=BATCH_SIZE)
            _, words = torch.max(probs, dim=2)
            words = words.cpu().data.numpy().tolist()
            for k in range(batch_size):
                hyp.append(tok.tokenize(tokenizer.decode(clean(words[k], tokenizer), skip_special_tokens=True, clean_up_tokenization_spaces=False).replace("' ", "'").lstrip()))
        print(hyp[:10])
    if fname is not None:
        positive_f = open("./%s/%s.1.txt"%(dname, fname), "w")
        negative_f = open("./%s/%s.0.txt"%(dname, fname), "w")
        hyp = [" ".join(x) + "\n" for x in hyp]
        length = int(len(hyp) / 2)
        if pos_num is not None:
            negative_f.writelines(hyp[:pos_num])
            positive_f.writelines(hyp[pos_num:])
        else:
            negative_f.writelines(hyp[:length])
            positive_f.writelines(hyp[length:])
        positive_f.close()
        negative_f.close()


def make_fasttest(addr1, addr2, addr3, fname="yelp"):
    """ data format: index, length, style """
    tokenizer = TreebankWordTokenizer()
    output = open("./fastText/" + addr3, "w")
    with open("./%s/"%fname + addr1, "r") as f:
        for x in f.readlines():
            output.write("__label__0 " + " ".join(tokenizer.tokenize(x.lower().strip())) + "\n")
    with open("./%s/"%fname + addr2, "r") as f:
        for x in f.readlines():
            output.write("__label__1 " + " ".join(tokenizer.tokenize(x.lower().strip())) + "\n")


def compute_fasttext(fname, dataset="yelp"):
    if dataset == "yelp":
        model_pt = "model_sentiment.bin"
    elif dataset == "amazon":
        model_pt = "model_amazon.bin"
    elif dataset == "imdb":
        model_pt = "model_imdb.bin"
    else:
        model_pt = "model_formality_family.bin"
    output = sp.check_output(["./fastText/fasttext", "test", f"./fastText/{model_pt}", f"./fastText/{fname}"], universal_newlines=True)
    output = output.split("\n")
    score = float(output[1].split("\t")[1])
    return score 


def ppl(sents, device, learned=True, dataset="yelp"):
    if learned:
        tokenizer = GPT2Tokenizer.from_pretrained('./%s/gpt'%dataset)
        model = GPT2LMHeadModel.from_pretrained('./%s/gpt'%dataset)
        tokenizer.bos_token = '<BOS>'
        tokenizer.pad_token = "<PAD>"
        print(tokenizer.add_tokens(['<negative>']))
        print(tokenizer.add_tokens(['<positive>']))
        print(tokenizer.add_tokens(['<PAD>']))
        print(tokenizer.add_tokens(['<BOS>']))
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(torch.load("./%s/result/language_model.pkl"%dataset))
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = model.to(device)
    cross_entropy = nn.CrossEntropyLoss()
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    with torch.no_grad():
        score = 0
        for (i, x) in enumerate(sents):
            if learned:
                tokens = [bos_id] + tokenizer.encode(x, add_special_tokens=False, add_prefix_space=True) + [eos_id]
            else:
                tokens = tokenizer.encode(x, add_special_tokens=False, add_prefix_space=True)
            if len(tokens) < 2:
                continue
            input_ids = torch.tensor(tokens).unsqueeze(0).cuda()  # Batch size 1
            outputs = model(input_ids, labels=input_ids)
            loss = outputs[0]
            score += loss.mean().item()
    return torch.exp(torch.tensor(score / i))


def evaluate_file_yelp(fname, device='cuda', learned=True, is_test=True):
    if is_test:
        # ref
        with open("./yelp/clean_reference.0.txt", "r") as f:
            lines = [x.strip() for x in f.readlines()]
        self_ref = [[x.split("\t")[0].split(" ")] for x in lines]
        ref = [[x.split("\t")[1].split(" ")] for x in lines]
        with open("./yelp/clean_reference.1.txt", "r") as f:
            lines = [x.strip() for x in f.readlines()]
        self_ref += [[x.split("\t")[0].split(" ")] for x in lines]
        ref += [[x.split("\t")[1].split(" ")] for x in lines]
    else:
        with open("./yelp/sentiment.dev.0.txt", "r") as f:
            self_ref = [[x.strip().split(" ")] for x in f.readlines()]
        with open("./yelp/sentiment.dev.1.txt", "r") as f:
            self_ref += [[x.strip().split(" ")] for x in f.readlines()]
    # hyp
    with open("./yelp/%s.0.txt"%fname, "r") as f:
        hyp = [x.strip().split(" ") for x in f.readlines()]
    with open("./yelp/%s.1.txt"%fname, "r") as f:
        hyp += [x.strip().split(" ") for x in f.readlines()]
    method = nltk.translate.bleu_score.SmoothingFunction(0.000000001).method1
    # print(ref[0])
    print(hyp[0])
    print(self_ref[0])
    result = dict()
    if is_test:
        print(len(self_ref))
        print(len(hyp))
        self_bleu = s_bleu(self_ref, hyp, method)
        bleu = s_bleu(ref, hyp, method)
        print("self bleu: %.6f"%(self_bleu))
        print("belu: %.6f"%(bleu))
        result["self_bleu"] = self_bleu
        result["bleu"] = bleu
        # sim
        self_ref = [" ".join(x[0]) for x in self_ref]
        ref = [" ".join(x[0]) for x in ref]
        hyp = [" ".join(x) for x in hyp]
        self_sim = sim_score_new(self_ref, hyp)
        sim = sim_score_new(ref, hyp)
        print("self sim: %.6f"%(self_sim))
        print("sim: %.6f"%(sim))
        result["self_sim"] = self_sim
        result["sim"] = sim 
    else:
        self_bleu = s_bleu(self_ref, hyp, method)
        print("self bleu: %.6f"%(self_bleu))
        result["self_bleu"] = self_bleu
        # sim
        self_ref = [" ".join(x[0]) for x in self_ref]
        hyp = [" ".join(x) for x in hyp]
        self_sim = sim_score_new(self_ref, hyp)
        print("self sim: %.6f"%(self_sim))
        result["self_sim"] = self_sim
    # PPL
    sents = []
    with open("./yelp/%s.0.txt"%fname, "r") as f:
        lines = f.readlines()
        sents = [x.strip() for x in lines]
    with open("./yelp/%s.1.txt"%fname, "r") as f:
        lines = f.readlines()
        sents += [x.strip() for x in lines]
    ppl_score = ppl(sents, device, learned)
    print("PPL: %.6f"%ppl_score)
    result["ppl"] = ppl_score
    # write fasttext file
    make_fasttest("%s.1.txt"%fname, "%s.0.txt"%fname, "%s"%fname)
    acc = compute_fasttext(fname)
    print(f"acc: {acc}")
    result["acc"] = acc
    return result


def evaluate_file_amazon(fname, device='cuda', learned=True, is_test=True):
    # ref
    if is_test:
        with open("./amazon/clean_reference.0.txt", "r") as f:
            lines = [x.strip() for x in f.readlines()]
        self_ref = [[x.split("\t")[0].split(" ")] for x in lines]
        ref = [[x.split("\t")[1].split(" ")] for x in lines]
        with open("./amazon/clean_reference.1.txt", "r") as f:
            lines = [x.strip() for x in f.readlines()]
        self_ref += [[x.split("\t")[0].split(" ")] for x in lines]
        ref += [[x.split("\t")[1].split(" ")] for x in lines]
    else:
        with open("./amazon/sentiment.dev.0.txt", "r") as f:
            self_ref = [[x.strip().split(" ")] for x in f.readlines()]
        with open("./amazon/sentiment.dev.1.txt", "r") as f:
            self_ref += [[x.strip().split(" ")] for x in f.readlines()]
    # hyp
    with open("./amazon/%s.0.txt"%fname, "r") as f:
        hyp = [x.strip().split(" ") for x in f.readlines()]
    with open("./amazon/%s.1.txt"%fname, "r") as f:
        hyp += [x.strip().split(" ") for x in f.readlines()]
    result = dict()
    method = nltk.translate.bleu_score.SmoothingFunction(0.000000001).method1
    print(self_ref[0])
    print(hyp[0])
    if is_test:
        self_bleu = s_bleu(self_ref, hyp, method)
        bleu = s_bleu(ref, hyp, method)
        print("self bleu: %.6f"%(self_bleu))
        print("belu: %.6f"%(bleu))
        result["self_bleu"] = self_bleu
        result["bleu"] = bleu
        # sim
        self_ref = [" ".join(x[0]) for x in self_ref]
        hyp = [" ".join(x) for x in hyp]
        ref = [" ".join(x[0]) for x in ref]
        self_sim = sim_score_new(self_ref, hyp)
        sim = sim_score_new(ref, hyp)
        result["self_sim"] = self_sim
        result["sim"] = sim 
        print("self sim: %.6f"%(self_sim))
        print("sim: %.6f"%(sim))
    else:
        self_bleu = s_bleu(self_ref, hyp, method)
        print("self bleu: %.6f"%(self_bleu))
        result["self_bleu"] = self_bleu
        # sim
        self_ref = [" ".join(x[0]) for x in self_ref]
        hyp = [" ".join(x) for x in hyp]
        self_sim = sim_score_new(self_ref, hyp)
        print("self sim: %.6f"%(self_sim))
        result["self_sim"] = self_sim
    # PPL
    sents = []
    with open("./amazon/%s.0.txt"%fname, "r") as f:
        lines = f.readlines()
        sents = [x.strip() for x in lines]
    with open("./amazon/%s.1.txt"%fname, "r") as f:
        lines = f.readlines()
        sents += [x.strip() for x in lines]
    ppl_score = ppl(sents, device, learned, dataset="amazon")
    result["ppl"] = ppl_score
    print("PPL: %.6f"%ppl_score)
    # write fasttext file
    make_fasttest("%s.1.txt"%fname, "%s.0.txt"%fname, "%s"%fname, "amazon")
    acc = compute_fasttext(fname, "amazon")
    print(f"acc: {acc}")
    result["acc"] = acc
    return result


def evaluate_file_imdb(fname, device='cuda', learned=True, is_test=True):
    # ref
    if is_test:
        with open("./imdb/sentiment.test.0.txt", "r") as f:
            self_ref = [[x.strip().split(" ")] for x in f.readlines()]
        with open("./imdb/sentiment.test.1.txt", "r") as f:
            self_ref += [[x.strip().split(" ")] for x in f.readlines()]
    else:
        with open("./imdb/sentiment.dev.0.txt", "r") as f:
            self_ref = [[x.strip().split(" ")] for x in f.readlines()]
        with open("./imdb/sentiment.dev.1.txt", "r") as f:
            self_ref += [[x.strip().split(" ")] for x in f.readlines()]
    # hyp
    with open("./imdb/%s.0.txt"%fname, "r") as f:
        hyp = [x.strip().split(" ") for x in f.readlines()]
    with open("./imdb/%s.1.txt"%fname, "r") as f:
        hyp += [x.strip().split(" ") for x in f.readlines()]
    result = dict()
    method = nltk.translate.bleu_score.SmoothingFunction(0.000000001).method1
    print(self_ref[0])
    print(hyp[0])
    self_bleu = s_bleu(self_ref, hyp, method)
    print("self bleu: %.6f"%(self_bleu))
    result["self_bleu"] = self_bleu
    # sim
    self_ref = [" ".join(x[0]) for x in self_ref]
    hyp = [" ".join(x) for x in hyp]
    self_sim = sim_score_new(self_ref, hyp)
    print("self sim: %.6f"%(self_sim))
    result["self_sim"] = self_sim
    # PPL
    ppl_score = ppl(hyp, device, learned, dataset="imdb")
    print("PPL: %.6f"%ppl_score)
    result["ppl"] = ppl_score
    # write fasttext file
    make_fasttest("%s.1.txt"%fname, "%s.0.txt"%fname, "%s"%fname, "imdb")
    acc = compute_fasttext(fname, "imdb")
    print(f"acc: {acc}")
    result["acc"] = acc
    return result


def evaluate_file_formality(fname, device='cuda', learned=True, is_test=True):
    dname = "./formality_family"
    # ref
    if is_test:
        with open("./%s/formality-gpt.test.json"%dname, "r") as f:
            data = json.load(f)
        self_ref = [[tok.tokenize(x["sent"])] for x in data]
        ref = [[tok.tokenize(s) for s in x["ref"]] for x in data]
    else:
        with open("./%s/formality-gpt.dev.json"%dname, "r") as f:
            data = json.load(f)
        self_ref = [[tok.tokenize(x["sent"])] for x in data]
    # hyp
    with open("./%s/%s.0.txt"%(dname, fname), "r") as f:
        hyp = [x.strip().split(" ") for x in f.readlines()]
    with open("./%s/%s.1.txt"%(dname, fname), "r") as f:
        hyp += [x.strip().split(" ") for x in f.readlines()]
    method = nltk.translate.bleu_score.SmoothingFunction(0.000000001).method1
    print(self_ref[0])
    print(hyp[0])
    result = dict()
    if is_test:
        self_bleu = s_bleu(self_ref, hyp, method)
        bleu = s_bleu(ref, hyp, method)
        print("self bleu: %.6f"%(self_bleu))
        print("belu: %.6f"%(bleu))
        result["self_bleu"] = self_bleu
        result["bleu"] = bleu
        # sim
        self_ref = [" ".join(x[0]) for x in self_ref]
        hyp = [" ".join(x) for x in hyp]
        self_sim = sim_score_new(self_ref, hyp)
        sim = 0
        for i in range(4):
            _ref = [" ".join(x[i]) for x in ref]
            sim += sim_score_new(_ref, hyp)
        sim /= 4
        print("self sim: %.6f"%(self_sim))
        print("sim: %.6f"%(sim))
        result["self_sim"] = self_sim
        result["sim"] = sim
    else:
        self_bleu = s_bleu(self_ref, hyp, method)
        print("self bleu: %.6f"%(self_bleu))
        result["self_bleu"] = self_bleu
        # sim
        self_ref = [" ".join(x[0]) for x in self_ref]
        hyp = [" ".join(x) for x in hyp]
        self_sim = sim_score_new(self_ref, hyp)
        print("self sim: %.6f"%(self_sim))
        result["self_sim"] = self_sim
    # PPL
    sents = []
    with open("./%s/%s.0.txt"%(dname, fname), "r") as f:
        lines = f.readlines()
        sents = [x.strip() for x in lines]
    with open("./%s/%s.1.txt"%(dname, fname), "r") as f:
        lines = f.readlines()
        sents += [x.strip() for x in lines]
    ppl_score = ppl(sents, device, learned, dataset=dname)
    print("PPL: %.6f"%ppl_score)
    result["ppl"] = ppl_score
    # write fasttext file
    make_fasttest("%s.1.txt"%fname, "%s.0.txt"%fname, "%s"%fname, dname)
    acc = compute_fasttext(fname, dname)
    print(f"acc: {acc}")
    result["acc"] = acc
    return result


def evaluation(args):
    # start experiment
    batch_size = 1
    fname = args.file
    DATASET = args.dataset
    model_pt = args.model_pt
    method = nltk.translate.bleu_score.SmoothingFunction(0.000000001).method1
    # load data
    tokenizer = GPT2Tokenizer.from_pretrained("./%s/gpt"%DATASET)
    tokenizer.bos_token = '<BOS>'
    tokenizer.pad_token = "<PAD>"
    tokenizer.add_tokens(['<negative>'])
    tokenizer.add_tokens(['<positive>'])
    tokenizer.add_tokens(['<PAD>'])
    tokenizer.add_tokens(['<BOS>'])
    if DATASET == "formality_family":
        with open("./%s/formality-gpt.test.json"%(DATASET), "r") as f:
            data = json.load(f)
    else:
        with open("./%s/sentiment-gpt.test.json"%(DATASET), "r") as f:
            data = json.load(f)
    if DATASET != "imdb":
        test_data = Dataloader.GPTRefLoader(data, tokenizer, batch_size, args.cuda)
    else:
        test_data = Dataloader.GPTLoader(data, tokenizer, batch_size, args.cuda)
    # build model
    generator = GPT2LMHeadModel.from_pretrained("./%s/gpt"%DATASET)
    generator.resize_token_embeddings(len(tokenizer))
    if args.model_pt is not None:
        generator.load_state_dict(torch.load(args.model_pt))
    if args.cuda:
        generator = generator.cuda()
    generator.eval()
    generate_output(generator, args, test_data, tokenizer, BATCH_SIZE=batch_size, fname=fname, dname=DATASET)
    if DATASET == "yelp":
        result = evaluate_file_yelp(fname, torch.device('cuda:%d'%args.gpuid), learned=False, is_test=True)
    elif DATASET == "amazon":
        result = evaluate_file_amazon(fname, torch.device('cuda:%d'%args.gpuid), learned=False, is_test=True)
    elif DATASET == "imdb":
        result = evaluate_file_imdb(fname, torch.device('cuda:%d'%args.gpuid), learned=False, is_test=True)
    else:
        result = evaluate_file_formality(fname, torch.device('cuda:%d'%args.gpuid), learned=False, is_test=True)
    print(result)
    return


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--file", default="", type=str)
    parser.add_argument("--dataset", default="", type=str)
    parser.add_argument("--model_pt", default=None, type=str)
    args = parser.parse_args()
    if args.cuda is False:
        evaluation(args)
    else:
        with torch.cuda.device(args.gpuid):
            evaluation(args)