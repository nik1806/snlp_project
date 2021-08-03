#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-02-23 01:52:54
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-03-18 01:32:15

"""
<Function of script>
"""

import os
import sys
import pdb
import re
import argparse
import string
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# 1 Data Preparation
def remove_emoticons(text):
    emoji_pattern = re.compile(
        u'([\U0001F1E6-\U0001F1FF]{2})|' # flags
        u'([\U0001F600-\U0001F64F])'     # emoticons
        "+", flags=re.UNICODE)
    return emoji_pattern.sub('', text)

def dataset_split(text):
    train, test = train_test_split(text, test_size=0.2, random_state=14)
    return train, test
    
def data_preparation(text, args, test_size:float=0.2):
    """ pre-process text corpus, split into train-test and save"""
    # Remove all punctuations except "? , !"
    remove = string.punctuation
    remove = remove.replace("?", "").replace(",", "").replace("!", "")
    pattern = r"[{}]".format(remove) # create the pattern
    text = re.sub(pattern, "", text) 
    # text = text.translate(str.maketrans('', '', string.punctuation))

    all_lines = text.split('\n')
    # Strip emoticons, english strings, whitespaces
    prepro_text = []
    for line in all_lines:
        if line:
            line = remove_emoticons(line) # remove flag symbols
            line = re.sub("[A-Za-z]+","",line) # remove english chars
            line = re.sub(r"\s+", " ", line) # remove white spaces
            line = re.sub(r'(\W)(?=\1)', '', line) # remove repeated occurrences of punctuation to single occurrence.
            line = line.replace("?", "?\n").replace(",", ",\n").replace("!", "!\n").replace("।", "।\n")
            lines = line.split("\n")
            for line in lines:
                prepro_text.append(line.strip())
            
    with open(args.out_dir + '/original_dataset.txt', 'w') as fpw:
        fpw.write('\n'.join(prepro_text))

    train, test = dataset_split(prepro_text)
    with open(args.out_dir + '/bn_train.txt', 'w') as train_fpw, open(args.out_dir + '/bn_test.txt', 'w') as test_fpw:
        train_fpw.write('\n'.join(train))
        test_fpw.write('\n'.join(test))


def vocab_generator(data):
    # pdb.set_trace()
    vocabs = set(data.split())
    return list(vocabs)

    
# def data_preparation(text, args):
#     """ pre-process text corpus, split into train-test and save"""
#     all_lines = text.split('\n')
#     # Strip emoticons, english strings, whitespaces
#     prepro_text = []
#     for line in all_lines:
#         if line:
#             line = remove_emoticons(line)       # remove flag symbols
#             line = re.sub("[A-Za-z]+","",line)  # remove english chars
#             line = re.sub(r"\s+", " ", line)    # remove white spaces
#             prepro_text.append(line.strip())
            
#     with open(args.out_dir + '/original_dataset.txt', 'w') as fpw:
#         fpw.write('\n'.join(prepro_text))

#     train, test = dataset_split(prepro_text)
#     with open(args.out_dir + '/bn_train.txt', 'w') as train_fpw, open(args.out_dir + '/bn_test.txt', 'w') as test_fpw:
#         train_fpw.write('\n'.join(train))
#         test_fpw.write('\n'.join(test))
    
# 2  Creating data for LM
def model_train(vocab_size, prefix, args):
    splits = ["train", "test"]
    for split in splits:
        input = str(args.out_dir + '/bn_' + split + '.txt')
        pref = str(args.spm_model_dir + '/' + prefix + '_' + split)
        
        # Step 1: train the model and specify the target vocabulary size
        cmd_step1 = "spm_train \
                    --input=" +  input + " \
                    --model_prefix=" + pref + " \
                    --vocab_size=" + str(vocab_size) + "     \
                    --character_coverage=0.995 \
                    --model_type=bpe"
        
        # Step 2: segment the text using this model.
        spm_model = pref + '.model'
        output = str(args.out_dir + '/'+ prefix + '_' + split + '.txt')
        cmd_step2 = "spm_encode \
                    --model=" + spm_model + " \
                    --output_format=piece \
                    < " + input + " \
                    > " + output
        os.system(cmd_step1)
        os.system(cmd_step2)

# 3 LM Training
def lm_train(prefix, class_size, args):
    # train language model based on the different subword 
    # granularity: char, subword unit(small vocab), subword unit(large vocab)
    os.makedirs(args.rnn_model_dir, exist_ok=True)
    print("*"*20 + " Begin LM training: " + "*"*20)
    train = str(args.out_dir + '/'+ prefix + '_' + "train" + '.txt')
    valid = str(args.out_dir + '/'+ prefix + '_' + "test" + '.txt')
    hid = str(args.hid)
    bptt = str(args.bptt)
    ext = prefix + '_hid_' + hid + "_bptt_" + bptt + "_class_" + class_size
    rnn_model = str(args.rnn_model_dir+'/rnnlm_'+ext)
    
    cmd = "rnnlm/rnnlm \
            -train " + train + " \
            -valid " + valid + " \
            -rnnlm " + rnn_model + " \
            -hidden " + hid + " \
            -rand-seed 1 \
            -debug 2 \
            -bptt " + bptt + " \
            -class " + class_size
    os.system(cmd)
    print("Model saved: " + rnn_model)
    print("*"*20 + " LM training DONE " + "*"*20)

# 4. Text Generation
def text_generation(prefix, class_size, args):
    # go from subword unit segmentation back to the original text 
    
    hid = str(args.hid)
    bptt = str(args.bptt)
    ext = prefix + '_hid_' + hid + "_bptt_" + bptt + "_class_" + class_size
    rnn_model =  str(args.rnn_model_dir+'/rnnlm_'+ext)
    spm_model = str(args.spm_model_dir + '/' + prefix + '_train.model')
    
    for k in range(1, 8):
        os.makedirs('gen_text_bn/'+ ext, exist_ok=True)
        cmd = "rnnlm/rnnlm \
            -rnnlm " + rnn_model + " \
            -gen " + str(10**k) + " \
            -debug 0 \
            >> " + str('gen_text_bn/' + ext + '/k_' + str(k) + '.txt')
            
        cmd_decode = "spm_decode \
            --model=" + spm_model + " \
            --input_format=piece \
            < " + str('gen_text_bn/' + ext + '/k_' + str(k) + '.txt') + " \
            > " + str('gen_text_bn/' + ext + '/k_' + str(k) + '_decoded.txt')
        
        os.system(cmd)
        os.system(cmd_decode)
        
    print("*"*20 + " Text generation completed " + "*"*20)
    
def oov_calculator(train, test):
    unseen_tokens = [tok for tok in test if tok not in train]
    oov_rate = len(unseen_tokens)/len(test)
    return oov_rate

# 5. OOV comparison
def oov_comparison(args):
    print("*"*10," OOV comparison ","*"*10)
    train_vocabs = vocab_generator(Path(args.out_dir + "/bn_train.txt").open('r').read())
    test_vocabs = vocab_generator(Path(args.out_dir + "/bn_test.txt").open('r').read())
    rnn_vocabs = defaultdict()
    oov_rates = []
    for path in Path('gen_text_bn').rglob('*decoded.txt'):
        k = path.name.split(".")[0]
        rnn_vocabs[k] = vocab_generator(Path(path).open('r').read())
        # 5.2
        print("Computing OOV rate: k =", k.split("_")[1], end = "\r")
        oov_rates.append(oov_calculator(train_vocabs+rnn_vocabs[k], test_vocabs))
        
    all_rnn_vocabs = list(set([word for item in list(rnn_vocabs.values()) for word in item]))
    # 5.1
    
    print("Generating OOV on all vocabs")
    all_vocabs = train_vocabs + all_rnn_vocabs
    oov_all_vocabs = oov_calculator(all_vocabs, test_vocabs)
    print("OOV rate (adding all rnn generated words to train vocab): %.3f" %(oov_all_vocabs))
    oov2 = "OOV rate for each K"
    with open(str(path).split("k_")[0] + '/oov.txt', 'w') as fpw:
        fpw.write('OOV rate (adding all rnn generated words to train vocab):'+str(oov_all_vocabs)+'\n')
        fpw.write(oov2+'\n')
        for k, ovr in enumerate(oov_rates):
            fpw.write(str(k)+':'+'\t'+str(ovr)+  '\n')
    
    # 5.3
    # if error, install 
    # sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
    xticklabels = []
    xaxis = []
    for i in range(1,8):
        xaxis.append(10**i)
        xticklabels.append(('$10^{'  +str(i) + '}$'))
    fig, ax = plt.subplots()
    plt.plot(xaxis, oov_rates)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(r'\textbf{K}')
    ax.set_ylabel(r'\textbf{OOV rate}')
    ax.set_title(r'\textbf{OOV: Bengali corpus}')
    plt.savefig(str(path).split("k_")[0] + '/bengali_oov.pdf', bbox_inches='tight')
    

def main():
    """ main method """
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.spm_model_dir, exist_ok=True)
    data_preparation(Path(args.corpus_fp).open('r').read(), args)
    vocab_sizes = [args.vocab_sizes]
    for prefix, vocab_size in enumerate(vocab_sizes):
        class_size = str(args.class_sizes)
        prefix = "bn_s" + str(prefix+1) + "_vocab_size_" + str(vocab_size)
        model_train(vocab_size, prefix, args)
        lm_train(prefix, class_size, args)
        text_generation(prefix, class_size, args)
        oov_comparison(args)

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_fp", help="path to text copus file")
    parser.add_argument("out_dir", help="path to save output files") 
    parser.add_argument("spm_model_dir", help="path to save spm trained models")
    parser.add_argument("rnn_model_dir", help="path to save rnn trained models") 
    parser.add_argument("-hid", default=40, type=int, help='hidden layer size')
    parser.add_argument("-bptt", default=3, type=int, help='num of steps to propagate error back ')
    parser.add_argument("-vocab_sizes", default=70, type=int, help='class_size')
    parser.add_argument("-class_sizes", default=70, type=int, help='class_size')
    # parser.add_argument("-class_sizes", nargs="+", default=[70, 100, 650], type=list, help='class size as list')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()
    
    