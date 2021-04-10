import classifier
import Dataloader
import argparse
from utils import start_log, myprint, isnan
import utils 
import time
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import evaluate
import nltk
from sim_models import WordAveraging
from sim_utils import Example
import sentencepiece as spm
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from gpt_utils import generate
from nltk.tokenize import TreebankWordTokenizer

config = {}
config["dataset"] = "formality_family"
config["BATCH_SIZE"] = 8
config["EPOCH"] = 200000
config["generator lr"] = 1e-5
config["discriminator lr"] = 5e-4
config["class lr"] = 5e-5
config["generator batch"] = 10
config["discriminator batch"] = 10
config["mle weight"] = 1
config["adv weight"] = 0.5
config["cycle weight"] = 1
config["sim weight"] = 20
config["language weight"] = 2
config["max_language_weight"] = 2
config["class weight"] = 2
config["grad clip"] = 1
config["g_dir"] = "../cache/919728/best/gen.dict"
config["goptim_dir"] = None
config["a_dir"] = "./%s/result/gpt_adv_classmodel.pkl"%config["dataset"]
config["aoptim_dir"] = None
config["b_dir"] = None
config["boptim_dir"] = None
config["mle_threshold"] = 0
config["cycle_threshold"] = 0
config["sim_threshold"] = 0
config["accumulation_step"] = 1
config["sentence_level"] = True
config["style_type"] = "formality"
config["acc_threshold"] = 0.9
STYLE_TYPE = config["style_type"]

DATASET = config["dataset"]
LOG = "-out.txt"
ID = config["id"] = random.randint(0, 1000000)

sp = spm.SentencePieceProcessor()
sp.Load('sim/sim.sp.30k.model')
tok = TreebankWordTokenizer()

def resume(id):
    fpath = utils.resume(id)
    config["g_dir"] = fpath["g_dir"]
    config["goptim_dir"] = fpath["goptim_dir"]
    config["a_dir"] = fpath["a_dir"]
    config["aoptim_dir"] = fpath["aoptim_dir"]
    config["b_dir"] = fpath["b_dir"]
    config["boptim_dir"] = fpath["boptim_dir"]

class Task():

    def __init__(self, tokenizer):
        self.source_dictionary = tokenizer
        self.target_dictionary = tokenizer

def get_mask(lengths, max_len):
    range_row = torch.arange(0, max_len).unsqueeze(0).expand(lengths.size(0), -1).long().type_as(lengths)
    lengths = lengths.unsqueeze(1).expand(-1, max_len)
    mask = range_row < lengths
    mask = mask.float()
    return mask

def make_example(sentence, model):
    sentence = sentence.lower()
    sentence = " ".join(tok.tokenize(sentence))
    sentence = sp.EncodeAsPieces(sentence)
    wp1 = Example(" ".join(sentence))
    wp1.populate_embeddings(model.vocab)
    return wp1

def pretrain_language_model(model, dataloader):
    model.train()
    goptimizer = optim.Adam(model.parameters(), lr=config["generator lr"])
    average_loss = 0
    for (i, batch) in enumerate(dataloader):
        # batch = dataloader.get()
        outputs = model(batch["src_text"], labels=batch["src_text"])
        mleloss = outputs[0]
        average_loss += mleloss.item()
        goptimizer.zero_grad()
        mleloss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        goptimizer.step()
        if (i + 1) % 100 == 0:
            # print("batch: %d, average loss: %.6f, loss: %.6f"%(i+1, average_loss, mleloss.item()))
            myprint("batch: %d, average loss: %.6f, loss: %.6f"%(i+1, average_loss / (i + 1), mleloss.item()))
    model.eval()
    model.cpu()
    torch.save(model.state_dict(), "./%s/result/language_model.pkl"%DATASET)
    model.cuda()

def compute_length_penalty(wl1, wl2, alpha=0.25):
    x = torch.stack((wl1.squeeze(), wl2.squeeze()), dim=1)
    x_min, _ = torch.min(x, dim=1)
    x_max, _ = torch.max(x, dim=1)
    ratio = x_max.float() / x_min.float()
    return torch.pow(torch.exp(1 - ratio.float()), alpha)

def main(args):
    # start experiment
    report_step = 100
    manualSeed = ID if args.seed == 0 else args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    if args.cuda:
        torch.cuda.manual_seed_all(manualSeed)
    string = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    start_log("./log/%d-"%ID + string + LOG, args.log)
    if args.resume != -1:
        resume(args.resume)
    myprint(args)
    for d in config:
        myprint("%s: %s" % (d, str(config[d])))
    args.batch_size = config["BATCH_SIZE"]

    # load data
    tokenizer = GPT2Tokenizer.from_pretrained("./%s/gpt"%DATASET)
    tokenizer.bos_token = '<BOS>'
    tokenizer.pad_token = "<PAD>"
    print(tokenizer.add_tokens(['<negative>']))
    print(tokenizer.add_tokens(['<positive>']))
    print(tokenizer.add_tokens(['<PAD>']))
    print(tokenizer.add_tokens(['<BOS>']))

    with open("./%s/%s-gpt.train.json"%(DATASET, STYLE_TYPE), "r") as f:
        data = json.load(f)
    dataloader = Dataloader.GPTLoader(data, tokenizer, args.batch_size, args.cuda, shuffle=True, input_maxlen=30)

    with open("./%s/%s-gpt.dev.json"%(DATASET, STYLE_TYPE), "r") as f:
        data = json.load(f)
    dev_data = Dataloader.GPTLoader(data, tokenizer, args.batch_size, args.cuda, shuffle=False)

    with open("./%s/%s-gpt.test.json"%(DATASET, STYLE_TYPE), "r") as f:
        data = json.load(f)
    if DATASET == "imdb":
        test_data = Dataloader.GPTLoader(data, tokenizer, args.batch_size, args.cuda)
    else:
        test_data = Dataloader.GPTRefLoader(data, tokenizer, args.batch_size, args.cuda)

    # build model
    generator = GPT2LMHeadModel.from_pretrained("./%s/gpt"%DATASET)
    generator.resize_token_embeddings(len(tokenizer))
    language_model = GPT2LMHeadModel.from_pretrained("./%s/gpt"%DATASET)
    language_model.resize_token_embeddings(len(tokenizer))
    language_model.load_state_dict(torch.load("./%s/result/language_model.pkl"%DATASET))
    language_model.eval()
    if config["g_dir"] is not None:
        generator.load_state_dict(torch.load(config["g_dir"]))
    discriminator_a = classifier.AdvDisNet(word_num=len(tokenizer))
    if config["a_dir"] is not None:
        discriminator_a.load_state_dict(torch.load(config["a_dir"]))

    discriminator_b = classifier.RNNDisNet(word_num=len(tokenizer), num_layers=1, dropout=0)
    sim_model = torch.load('sim/sim.pt', map_location='cpu')
    state_dict = sim_model['state_dict']
    vocab_words = sim_model['vocab_words']
    sim_args = sim_model['args']
    sim_args.gpu = args.gpuid
    sim_model = WordAveraging(sim_args, vocab_words)
    sim_model.load_state_dict(state_dict, strict=True)
    L = nn.CrossEntropyLoss()
    BL = nn.BCELoss()

    if args.cuda:
        generator = generator.cuda()
        discriminator_a = discriminator_a.cuda()
        discriminator_b = discriminator_b.cuda()
        sim_model = sim_model.cuda()
        L = L.cuda()
        BL = BL.cuda()
        language_model = language_model.cuda()
        if args.critic:
            critic = critic.cuda()

    goptimizer = optim.Adam(generator.parameters(), lr=config["generator lr"])
    if config["goptim_dir"] is not None:
        goptimizer.load_state_dict(torch.load(config["goptim_dir"], map_location=torch.device('cuda', args.gpuid)))
    for param_group in goptimizer.param_groups:
        param_group['lr'] = config["generator lr"]
    doptimizer_a = optim.Adam(discriminator_a.parameters(), lr=config["class lr"])
    doptimizer_b = optim.Adam(discriminator_b.parameters(), lr=config["discriminator lr"])
    if config["aoptim_dir"] is not None:
        doptimizer_a.load_state_dict(torch.load(config["aoptim_dir"], map_location=torch.device('cuda', args.gpuid)))
    for param_group in doptimizer_a.param_groups:
        param_group['lr'] = config["class lr"]

    
    EPOCH = config["EPOCH"]
    GBATCH = config["generator batch"]
    DBATCH = config["discriminator batch"]
    W_M = config["mle weight"]
    W_A = config["adv weight"]
    W_S = config["sim weight"]
    W_C = config["cycle weight"]
    W_L = config["language weight"]
    W_D = config["class weight"]
    GRAD_CLIP = config["grad clip"]
    PRETRAIN_BATCH = 0
    accumulation_step = config["accumulation_step"]

    gloss_all, gloss_mle, gloss_adv, gloss_cycle, gloss_sim, dloss_a, dloss_b, gcnt, dcnt, avg_language_loss, avg_language_score, avg_adv_score, avg_language_diff = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    avg_fake_loss, avg_real_loss, avg_sim_score, avg_critic_loss = 0, 0, 0, 0
    avg_cls_loss, avg_cls_score, gloss_class, avg_real_loss_cls, avg_fake_loss_cls = 0, 0, 0, 0, 0
    best_record = 1000
    if args.log:
        os.mkdir("./cache/%d"%(ID))
        os.mkdir("./cache/%d/best/"%(ID))
    best_gname = "./cache/%d/best/gen.dict" % ID
    best_a_dname = "./cache/%d/best/a_dis.dict" % ID
    best_b_dname = "./cache/%d/best/b_dis.dict" % ID
    best_goname = "./cache/%d/best/genopt.dict" % ID
    best_a_doname = "./cache/%d/best/a_disopt.dict" % ID
    best_b_doname = "./cache/%d/best/b_disopt.dict" % ID

    gscheduler = optim.lr_scheduler.StepLR(goptimizer, step_size=500, gamma=0.5)
    dscheduler = optim.lr_scheduler.StepLR(doptimizer_a, step_size=250, gamma=0.5)
    fine_tune_stage = args.reinforce
    language_loss_fct = nn.CrossEntropyLoss(reduce=False)
    prev_language_score = 0
    print(classifier.classifer_test(discriminator_a, tokenizer, dev_data, args.batch_size))
    one_tensor = torch.ones(1)
    if args.cuda:
        one_tensor = one_tensor.cuda() 
    # pretrain_language_model(language_model, dataloader)
    for i in range(EPOCH):
        # generator training
        generator.train()
        discriminator_a.eval()
        step_cnt = 0
        goptimizer.zero_grad()
        for j in range(GBATCH * accumulation_step):
            # print(gcnt)
            step_cnt += 1
            batch = dataloader.get()
            # reconstruction loss
            rec_text = torch.cat((batch["src_text"], batch["style_tokens"].unsqueeze(1), batch["src_text"]), dim=1)
            outputs = generator(rec_text, labels=rec_text)
            mleloss = outputs[0]
            mleloss_ = F.threshold(mleloss, config["mle_threshold"], 0)
            # classifier loss
            transfer_text = torch.cat((batch["src_text"], batch["transfer_tokens"].unsqueeze(1)), dim=1)
            cur_len = transfer_text.size(1)
            _, probs = generate(generator, transfer_text, cur_len=cur_len, max_length=int(cur_len * 2 - 1), pad_token_id=tokenizer.pad_token_id,
             eos_token_ids=tokenizer.eos_token_id, batch_size=args.batch_size)
            probs = F.softmax(probs, dim=2)
            idx_probs, words = torch.max(probs, dim=2)
            style_pred = discriminator_a.approximate(probs, 1 - batch["style"])
            style_pred = torch.squeeze(style_pred, 1)
            class_loss = - torch.log(style_pred + 0.0001).mean()
            # adv loss
            adv_pred = discriminator_b.approximate(probs)
            adv_pred = torch.squeeze(adv_pred, 1)
            advloss = - torch.log(adv_pred + 0.0001).mean()
            # sim loss
            if args.sim:
                wx1, wl1, wm1 = sim_model.torchify_batch([make_example(x, sim_model) for x in batch["tokens"]])
                words_ = words.cpu().data.numpy().tolist()
                generate_sents = [tokenizer.decode(evaluate.clean(sent, tokenizer), skip_special_tokens=True, clean_up_tokenization_spaces=False).replace("' ", "'").lstrip() for sent in words_]
                wx2, wl2, wm2 = sim_model.torchify_batch([make_example(x, sim_model) for x in generate_sents])
                with torch.no_grad():
                    sim_scores = sim_model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
                avg_sim_score += sim_scores.mean().item()
                if args.length_penalty:
                    length_penalty = compute_length_penalty(wl1, wl2, 0.25)
                else:
                    length_penalty = 1
                simloss = torch.mul(- torch.mul(sim_scores, length_penalty), torch.log(idx_probs).mean(dim=1)).mean()
            else:
                simloss = torch.zeros(1).cuda()

            # language fluency loss
            with torch.no_grad():
                outputs = language_model(words)
                true_outputs = language_model(batch["src_text"])
            lm_logits = outputs[0]
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = words[..., 1:].contiguous()
            language_loss = language_loss_fct(shift_logits.transpose(1, 2), shift_labels)
            lengths = torch.LongTensor([evaluate.get_len(x, tokenizer) for x in words_]) - 1
            lengths = lengths.cuda() if args.cuda else lengths
            mask = get_mask(lengths, language_loss.size(1))
            if config["sentence_level"]:
                language_loss = torch.mul(mask, language_loss).sum(1) / (lengths.float() + 0.001)
                true_lm_logits = true_outputs[0]
                true_shift_logits = true_lm_logits[..., :-1, :].contiguous()
                true_shift_labels = batch["src_text"][..., 1:].contiguous()
                true_language_loss = language_loss_fct(true_shift_logits.transpose(1, 2), true_shift_labels)
                true_lengths = batch["length"] - 1
                true_mask = get_mask(true_lengths, true_language_loss.size(1))
                true_language_loss = torch.mul(true_mask, true_language_loss).sum(1) / (true_lengths.float() + 0.001)
                avg_language_diff += (language_loss.mean() - true_language_loss.mean()).item()
            now_language_score = language_loss.mean().item()
            if config["sentence_level"]:
                language_loss = torch.mul(language_loss - true_language_loss, torch.mul(mask, torch.log(idx_probs[:, 1:])).sum(1) / (lengths.float() + 0.001)).mean()
            else:
                language_loss = (torch.mul(torch.mul(language_loss, torch.log(idx_probs[:, 1:])), mask).sum(1) / (lengths.float() + 0.001)).mean()
            avg_language_loss += language_loss.item()
            avg_language_score += now_language_score

            # compute loss
            if gcnt < PRETRAIN_BATCH:
                loss = W_M * mleloss_
            else:
                loss = W_M * mleloss_ + W_A * advloss + W_S * simloss + W_L * language_loss + W_D * class_loss
            gloss_all += loss.item() / accumulation_step
            gloss_mle += mleloss.item()
            gloss_adv += advloss.item()
            gloss_sim += simloss.item()
            gloss_class += class_loss.item()
            now_advloss = advloss.item()
            now_simloss = simloss.item()
            now_loss = loss.item()
            now_mleloss = mleloss.item()
            loss = loss / accumulation_step # normalizing
            loss.backward()
            if step_cnt % accumulation_step == 0:
                gcnt += 1
                step_cnt = 0
                nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)
                goptimizer.step()
                goptimizer.zero_grad()
                if W_L < config["max_language_weight"]:
                    # adjusting weights
                    W_L += 1
                del advloss, mleloss, mleloss_, loss, simloss
                torch.cuda.empty_cache()
        # discriminator training
        discriminator_b.train()
        discriminator_a.train()
        generator.eval()
        doptimizer_a.zero_grad()
        doptimizer_b.zero_grad()
        for j in range(DBATCH):
            if gcnt < PRETRAIN_BATCH:
                break
            batch = dataloader.get()
            transfer_text = torch.cat((batch["src_text"], batch["transfer_tokens"].unsqueeze(1)), dim=1)
            cur_len = transfer_text.size(1)
            with torch.no_grad():
            	_, probs = generate(generator, transfer_text, cur_len=cur_len, max_length=int(cur_len * 2 - 1), pad_token_id=tokenizer.pad_token_id,
            	 eos_token_ids=tokenizer.eos_token_id, batch_size=args.batch_size)
            	probs = F.softmax(probs, dim=2)
            	probs.detach_()
            # discriminator for naturalness
            if args.reinforce:
                probs, words = torch.max(probs, dim=2)
                style_pred = discriminator_b(words)
            else:
                style_pred = discriminator_b.approximate(probs)
            style_pred = torch.squeeze(style_pred, 1)
            real_style_pred_true = discriminator_b(batch["src_text"])
            real_style_pred_ture = torch.squeeze(real_style_pred_true, 1)
            fake_loss_b = - torch.log(1 - style_pred).mean()
            real_loss_b = - torch.log(real_style_pred_true).mean()
            advloss_b = real_loss_b + fake_loss_b
            avg_fake_loss += fake_loss_b.item()
            avg_real_loss += real_loss_b.item()
            now_fake_loss = fake_loss_b.item()
            now_real_loss = real_loss_b.item()
            now_dis_loss = advloss_b.item()
            dloss_b += advloss_b.item()
            doptimizer_b.zero_grad()
            advloss_b.backward()
            nn.utils.clip_grad_norm_(discriminator_b.parameters(), GRAD_CLIP)
            doptimizer_b.step()
            # discriminator for style
            if args.update_style:
                if args.reinforce:
                    style_pred = discriminator_a(words, 1 - batch["style"])
                else:
                    style_pred = discriminator_a.approximate(probs, 1 - batch["style"])
                style_pred = torch.squeeze(style_pred, 1)
                real_style_pred_true = discriminator_a(batch["src_text"], batch["style"])
                real_style_pred_ture = torch.squeeze(real_style_pred_true, 1)
                fake_loss_a = - torch.log(1 - style_pred).mean()
                real_loss_a = - torch.log(real_style_pred_true).mean()
                advloss_a = real_loss_a + fake_loss_a
                avg_fake_loss_cls += fake_loss_a.item()
                avg_real_loss_cls += real_loss_a.item()
                dloss_a += advloss_a.item()
                doptimizer_a.zero_grad()
                advloss_a.backward()
                nn.utils.clip_grad_norm_(discriminator_a.parameters(), GRAD_CLIP)
                doptimizer_a.step()
            else:
                real_loss_a = 0
                fake_loss_a = 0
                advloss_a = 0
            dcnt += 1
            del real_loss_b, fake_loss_b, advloss_b, real_loss_a, fake_loss_a, advloss_a
            torch.cuda.empty_cache()

        if gcnt % report_step == 0:
            myprint("task id: %d"%ID)
            myprint("generator training batch: %d"%gcnt)
            myprint("average loss: %.6f"%(gloss_all / report_step))
            myprint("average adv loss: %.6f"%(gloss_adv / (report_step * accumulation_step)))
            myprint("average mle loss: %.6f"%(gloss_mle / (report_step * accumulation_step)))
            myprint("average cycle loss: %.6f"%(gloss_cycle / (report_step * accumulation_step)))
            myprint("average sim loss: %.6f"%(gloss_sim / (report_step * accumulation_step)))
            myprint("average sim score: %.6f"%(avg_sim_score / (report_step * accumulation_step)))
            myprint("avg class loss: %.6f"%(gloss_class / (report_step * accumulation_step)))
            myprint("avg class score: %.6f"%(avg_cls_score / (report_step * accumulation_step)))
            myprint("avg language score: %.6f"%(avg_language_score  / (report_step * accumulation_step)))
            myprint("avg language loss: %.6f"%(avg_language_loss  / (report_step * accumulation_step)))
            if config["sentence_level"]:
                myprint("avg language diff: %.6f"%(avg_language_diff  / (report_step * accumulation_step)))
            myprint("avg adv score: %.6f"%(avg_adv_score / (report_step * accumulation_step)))
            avg_language_loss, avg_language_score, avg_adv_score, avg_language_diff = 0, 0, 0, 0
            myprint()
            gloss_all, gloss_mle, gloss_adv, gloss_cycle, gloss_sim, avg_sim_score, gloss_class, avg_cls_score = 0, 0, 0, 0, 0, 0, 0, 0

        if dcnt % report_step == 0 and dcnt != 0:
            myprint("discriminator training batch: %d"%dcnt)
            myprint("b average loss: %.6f"%(dloss_b / (report_step)))
            myprint("avg real loss: %.6f"%(avg_real_loss/(report_step)))
            myprint("avg fake loss: %.6f"%(avg_fake_loss/(report_step)))
            myprint("a average loss: %.6f"%(dloss_a / (report_step)))
            myprint("avg real cls loss: %.6f"%(avg_real_loss_cls/(report_step)))
            myprint("avg fake cls loss: %.6f"%(avg_fake_loss_cls/(report_step)))
            myprint()
            dloss_a, dloss_b, avg_real_loss, avg_fake_loss, avg_real_loss_cls, avg_fake_loss_cls = 0, 0, 0, 0, 0, 0

        gscheduler.step()
        dscheduler.step()
        string = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        gname = "./cache/%d/gen-%s.dict" % (ID, string)
        a_dname = "./cache/%d/a_dis-%s.dict" % (ID, string)
        b_dname = "./cache/%d/b_dis-%s.dict" % (ID, string)
        goname = "./cache/%d/genopt-%s.dict" % (ID, string)
        a_doname = "./cache/%d/a_disopt-%s.dict" % (ID, string)
        b_doname = "./cache/%d/b_disopt-%s.dict" % (ID, string)
        if gcnt % 1000 == 0 and args.log:
            generator.eval()
            result = test(generator, "dev")
            acc_transfer = result["acc"]
            self_bleu = result["self_bleu"]
            dev_acc = acc_transfer
            dev_bleu = self_bleu
            dev_ppl = result["ppl"]
            myprint(f"gcnt: {gcnt}")
            myprint("dev set:")
            myprint("acc transfer: %.6f"%acc_transfer)
            myprint("self_bleu: %.6f"%self_bleu)
            myprint("ppl: %.6f"%dev_ppl)
            result = test(generator, "test")
            acc_transfer = result["acc"]
            self_bleu = result["self_bleu"]
            ppl = result["ppl"]
            myprint("test set:")
            myprint("acc transfer: %.6f"%acc_transfer)
            myprint("self_bleu: %.6f"%self_bleu)
            myprint("ppl: %.6f"%ppl)
            if DATASET != "imdb":
                bleu = result["bleu"]
                myprint("bleu: %.6f"%bleu)
            generator.train()
            generator.cpu()
            discriminator_a.cpu()
            f_score = 2 * dev_acc * dev_bleu / (dev_acc + dev_bleu)
            if dev_ppl < best_record and dev_acc > config["acc_threshold"] and gcnt > PRETRAIN_BATCH:
                best_record = dev_ppl
                myprint("best")
                myprint("acc transfer: %.6f"%acc_transfer)
                myprint("self_bleu: %.6f"%self_bleu)
                myprint("ppl: %.6f"%ppl)
                if DATASET != "imdb":
                    myprint("bleu: %.6f"%bleu)
                myprint()
                torch.save(generator.state_dict(), best_gname)
                torch.save(discriminator_a.state_dict(), best_a_dname)
                torch.save(goptimizer.state_dict(), best_goname)
                torch.save(doptimizer_a.state_dict(), best_a_doname)
            if gcnt > PRETRAIN_BATCH:
                gname = "./cache/%d/gen-%d.dict" % (ID, gcnt)
                a_dname = "./cache/%d/a_dis-%d.dict" % (ID, gcnt)
                torch.save(generator.state_dict(), gname)
                torch.save(discriminator_a.state_dict(), a_dname)
            if args.cuda:
                generator.cuda()
                discriminator_a.cuda()


def test(generator, split):
    # start experiment
    generator.eval()
    batch_size = 1
    method = nltk.translate.bleu_score.SmoothingFunction(0.000000001).method1
    # load data
    tokenizer = GPT2Tokenizer.from_pretrained("./%s/gpt"%DATASET)
    tokenizer.bos_token = '<BOS>'
    tokenizer.pad_token = "<PAD>"
    tokenizer.add_tokens(['<negative>'])
    tokenizer.add_tokens(['<positive>'])
    tokenizer.add_tokens(['<PAD>'])
    tokenizer.add_tokens(['<BOS>'])
    fname = "tmp"
    if DATASET == "formality_family":
        with open("./%s/formality-gpt.%s.json"%(DATASET, split), "r") as f:
            data = json.load(f)
    else:
        with open("./%s/sentiment-gpt.%s.json"%(DATASET, split), "r") as f:
            data = json.load(f)
    if split == "test":
        if DATASET == "imdb":
            test_data = Dataloader.GPTLoader(data, tokenizer, batch_size, args.cuda)
        else:
            test_data = Dataloader.GPTRefLoader(data, tokenizer, batch_size, args.cuda)
        evaluate.generate_output(generator, args, test_data, tokenizer, BATCH_SIZE=batch_size, fname=fname, dname=DATASET)
        # do evaluation
        if DATASET == "yelp":
            result = evaluate.evaluate_file_yelp(fname, torch.device('cuda:%d'%args.gpuid), learned=False, is_test=True)
        elif DATASET == "amazon":
            result = evaluate.evaluate_file_amazon(fname, torch.device('cuda:%d'%args.gpuid), learned=False, is_test=True)
        elif DATASET == "imdb":
            result = evaluate.evaluate_file_imdb(fname, torch.device('cuda:%d'%args.gpuid), learned=False, is_test=True)
        else:
            result = evaluate.evaluate_file_formality(fname, torch.device('cuda:%d'%args.gpuid), learned=False, is_test=True)
    else:
        test_data = Dataloader.GPTLoader(data, tokenizer, batch_size, args.cuda)
        if DATASET == "amazon":
            evaluate.generate_output(generator, args, test_data, tokenizer, BATCH_SIZE=batch_size, fname=fname, dname=DATASET, pos_num=985)
        elif DATASET == "formality_family":
            evaluate.generate_output(generator, args, test_data, tokenizer, BATCH_SIZE=batch_size, fname=fname, dname=DATASET, pos_num=2247)
        else:
            evaluate.generate_output(generator, args, test_data, tokenizer, BATCH_SIZE=batch_size, fname=fname, dname=DATASET)
        # do evaluation
        if DATASET == "yelp":
            result = evaluate.evaluate_file_yelp(fname, torch.device('cuda:%d'%args.gpuid), learned=False, is_test=False)
        elif DATASET == "amazon":
            result = evaluate.evaluate_file_amazon(fname, torch.device('cuda:%d'%args.gpuid), learned=False, is_test=False)
        elif DATASET == "imdb":
            result = evaluate.evaluate_file_imdb(fname, torch.device('cuda:%d'%args.gpuid), learned=False, is_test=False)
        else:
            result = evaluate.evaluate_file_formality(fname, torch.device('cuda:%d'%args.gpuid), learned=False, is_test=False)
    generator.train()
    return result


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--resume", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("-l", "--log", action="store_true")
    parser.add_argument("-e", "--evaluate", action="store_true")
    parser.add_argument("-n", "--nocycle", action="store_true")
    parser.add_argument("-g", "--generate", action="store_true")
    parser.add_argument("-s", "--sim", action="store_true")
    parser.add_argument("-r", "--reinforce", action="store_true")
    parser.add_argument("-p", "--length_penalty", action="store_true")
    parser.add_argument("-u", "--update_style", action="store_true")    
    args = parser.parse_args()
    if args.cuda is False:
        main(args)
    else:
        with torch.cuda.device(args.gpuid):
            main(args)

