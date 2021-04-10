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
import pickle
import evaluate
import nltk
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from gpt_utils import generate

config = {}
config["BATCH_SIZE"] = 8
config["EPOCH"] = 200000
config["generator lr"] = 1e-5
config["discriminator lr"] = 1e-4
config["generator batch"] = 10
config["discriminator batch"] = 5
config["mle weight"] = 1
config["adv weight"] = 1
config["cycle weight"] = 2
config["grad clip"] = 1
config["dataset"] = "formality_family" 
config["g_dir"] = None
config["goptim_dir"] = None
config["a_dir"] = "./%s/result/gpt_adv_classmodel.pkl"%config["dataset"]
config["aoptim_dir"] = None
config["mle_threshold"] = 0
config["cycle_threshold"] = 0
config["accumulation_step"] = 1
config["style_type"] = "formality"
STYLE_TYPE = config["style_type"]

LOG = "-out.txt"
ID = config["id"] = random.randint(0, 1000000)
DATASET = config["dataset"]


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
    dataloader = Dataloader.GPTLoader(data, tokenizer, args.batch_size, args.cuda, shuffle=True, input_maxlen=40)

    with open("./%s/%s-gpt.dev.json"%(DATASET, STYLE_TYPE), "r") as f:
        data = json.load(f)
    dev_data = Dataloader.GPTLoader(data, tokenizer, args.batch_size, args.cuda, shuffle=False)

    with open("./%s/%s-gpt.test.json"%(DATASET, STYLE_TYPE), "r") as f:
        data = json.load(f)
    if DATASET != "imdb":
        test_data = Dataloader.GPTRefLoader(data, tokenizer, args.batch_size, args.cuda)
    else:
        test_data = Dataloader.GPTLoader(data, tokenizer, args.batch_size, args.cuda)

    # build model
    generator = GPT2LMHeadModel.from_pretrained("./%s/gpt"%DATASET)
    generator.resize_token_embeddings(len(tokenizer))
    if config["g_dir"] is not None:
        generator.load_state_dict(torch.load(config["g_dir"]))
    discriminator_a = classifier.AdvDisNet(word_num=len(tokenizer))
    if config["a_dir"] is not None:
        discriminator_a.load_state_dict(torch.load(config["a_dir"]))

    L = nn.CrossEntropyLoss()

    if args.cuda:
        generator = generator.cuda()
        discriminator_a = discriminator_a.cuda()
        L = L.cuda()

    goptimizer = optim.Adam(generator.parameters(), lr=config["generator lr"])
    if config["goptim_dir"] is not None:
        goptimizer.load_state_dict(torch.load(config["goptim_dir"], map_location=torch.device('cuda', args.gpuid)))
    for param_group in goptimizer.param_groups:
        param_group['lr'] = config["generator lr"]
    doptimizer_a = optim.Adam(discriminator_a.parameters(), lr=config["discriminator lr"])
    if config["aoptim_dir"] is not None:
        doptimizer_a.load_state_dict(torch.load(config["aoptim_dir"], map_location=torch.device('cuda', args.gpuid)))
    for param_group in doptimizer_a.param_groups:
        param_group['lr'] = config["discriminator lr"]
    
    EPOCH = config["EPOCH"]
    GBATCH = config["generator batch"]
    DBATCH = config["discriminator batch"]
    W_M = config["mle weight"]
    W_A = config["adv weight"]
    W_C = config["cycle weight"]
    GRAD_CLIP = config["grad clip"]
    PRETRAIN_BATCH = 1000
    accumulation_step = config["accumulation_step"]

    gloss_all, gloss_mle, gloss_adv, gloss_cycle, dloss_a, gcnt, dcnt, avg_adv_score = 0, 0, 0, 0, 0, 0, 0, 0
    avg_fake_loss, avg_real_loss = 0, 0
    best_record = 0
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
    for i in range(EPOCH):
        # generator training
        generator.train()
        discriminator_a.eval()
        step_cnt = 0
        goptimizer.zero_grad()
        for j in range(GBATCH * accumulation_step):
            step_cnt += 1
            print(gcnt)
            batch = dataloader.get()
            # reconstruction loss
            rec_text = torch.cat((batch["src_text"], batch["style_tokens"].unsqueeze(1), batch["src_text"]), dim=1)
            outputs = generator(rec_text, labels=rec_text)
            mleloss = outputs[0]
            mleloss_ = F.threshold(mleloss, config["mle_threshold"], 0)
            # style classifier loss
            transfer_text = torch.cat((batch["src_text"], batch["transfer_tokens"].unsqueeze(1)), dim=1)
            cur_len = transfer_text.size(1)
            _, probs = generate(generator, transfer_text, cur_len=cur_len, max_length=int(cur_len * 2 - 1), pad_token_id=tokenizer.pad_token_id,
             eos_token_ids=tokenizer.eos_token_id, batch_size=args.batch_size)
            probs = F.softmax(probs, dim=2)
            idx_probs, words = torch.max(probs, dim=2)
            style_pred = discriminator_a.approximate(probs, 1 - batch["style"])
            style_pred = torch.squeeze(style_pred, 1)
            advloss = - torch.log(style_pred + 0.0001).mean()

            # cycle loss
            if args.nocycle:
                cycleloss = torch.zeros(1)
                cycleloss_ = 0
            else:
                cur_len = batch["src_text"].size(1) + 1
                _, probs = generate(generator, probs, cur_len=cur_len, max_length=int(cur_len * 2 - 1), pad_token_id=tokenizer.pad_token_id,
                 eos_token_ids=tokenizer.eos_token_id, batch_size=args.batch_size, approximate=True, style_token=batch["style_tokens"])
                probs = probs.transpose(1, 2)
                cycleloss = L(probs, batch["src_text"])
                cycleloss_ = F.threshold(cycleloss, config["cycle_threshold"], 0)

            if gcnt < PRETRAIN_BATCH:
                loss = W_M * mleloss_
            else:
                loss = W_M * mleloss_ + W_A * advloss + W_C * cycleloss_
            gloss_all += loss.item() / accumulation_step
            gloss_mle += mleloss.item()
            gloss_adv += advloss.item()
            gloss_cycle += cycleloss.item()
            loss = loss / accumulation_step # normalizing
            loss.backward()
            if step_cnt % accumulation_step == 0:
                gcnt += 1
                step_cnt = 0
                nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)
                goptimizer.step()
                goptimizer.zero_grad()
                del advloss, cycleloss, mleloss, mleloss_, loss, cycleloss_
                torch.cuda.empty_cache()
        # discriminator training
        discriminator_a.train()
        generator.eval()
        doptimizer_a.zero_grad()
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
            style_pred = discriminator_a.approximate(probs,  1 - batch["style"])
            style_pred = torch.squeeze(style_pred, 1)
            real_style_pred_true = discriminator_a(batch["src_text"], batch["style"])
            real_style_pred_false = discriminator_a(batch["src_text"], 1 - batch["style"])
            real_style_pred_ture = torch.squeeze(real_style_pred_true, 1)
            real_style_pred_false = torch.squeeze(real_style_pred_false, 1)
            fake_loss_a = - torch.log(1 - style_pred).mean()
            real_loss_a = (- torch.log(real_style_pred_true).mean() - torch.log(1 - real_style_pred_false).mean()) / 2
            advloss_a = real_loss_a  + fake_loss_a
            avg_fake_loss += fake_loss_a.item()
            avg_real_loss += real_loss_a.item()
            now_fake_loss = fake_loss_a.item()
            now_real_loss = real_loss_a.item()
            now_dis_loss = advloss_a.item()
            dloss_a += advloss_a.item()
            doptimizer_a.zero_grad()
            advloss_a.backward()
            nn.utils.clip_grad_norm_(discriminator_a.parameters(), GRAD_CLIP)
            doptimizer_a.step()
            dcnt += 1
            del real_loss_a, fake_loss_a, advloss_a
            torch.cuda.empty_cache()

        if gcnt % report_step == 0 and gcnt != 0:
            myprint("task id: %d"%ID)
            myprint("generator training batch: %d"%gcnt)
            myprint("average loss: %.6f"%(gloss_all / report_step))
            myprint("average adv loss: %.6f"%(gloss_adv / (report_step * accumulation_step)))
            myprint("average mle loss: %.6f"%(gloss_mle / (report_step * accumulation_step)))
            myprint("average cycle loss: %.6f"%(gloss_cycle / (report_step * accumulation_step)))
            myprint()
            gloss_all, gloss_mle, gloss_adv, gloss_cycle = 0, 0, 0, 0

        if dcnt % report_step == 0 and dcnt != 0:
            myprint("discriminator training batch: %d"%dcnt)
            myprint("a average loss: %.6f"%(dloss_a / (report_step)))
            myprint("avg real loss: %.6f"%(avg_real_loss/(report_step)))
            myprint("avg fake loss: %.6f"%(avg_fake_loss/(report_step)))
            myprint()
            dloss_a, avg_real_loss, avg_fake_loss = 0, 0, 0

        gscheduler.step()
        dscheduler.step()
        if gcnt % 1000 == 0 and args.log:
            generator.eval()
            result = test(generator, "dev")
            acc_transfer = result["acc"]
            self_bleu = result["self_bleu"]
            dev_acc = acc_transfer
            dev_bleu = self_bleu
            myprint(f"gcnt: {gcnt}")
            myprint("dev set:")
            myprint("acc transfer: %.6f"%acc_transfer)
            myprint("self_bleu: %.6f"%self_bleu)
            result = test(generator, "test")
            acc_transfer = result["acc"]
            self_bleu = result["self_bleu"]
            if DATASET != "imdb":
                bleu = result["bleu"]
            myprint("test set:")
            myprint("acc transfer: %.6f"%acc_transfer)
            myprint("self_bleu: %.6f"%self_bleu)
            if DATASET != "imdb":
                myprint("bleu: %.6f"%bleu)
            generator.train()
            generator.cpu()
            discriminator_a.cpu()
            f_score = 2 * dev_acc * dev_bleu / (dev_acc + dev_bleu)
            if f_score > best_record and gcnt > PRETRAIN_BATCH:
                best_record = f_score
                myprint("best")
                myprint("acc transfer: %.6f"%acc_transfer)
                myprint("self_bleu: %.6f"%self_bleu)
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
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("-l", "--log", action="store_true")
    parser.add_argument("-n", "--nocycle", action="store_true")
    args = parser.parse_args()
    if args.cuda is False:
        main(args)
    else:
        with torch.cuda.device(args.gpuid):
            main(args)
    
