# dependencies
import string
from typing import List, Dict
import nltk.data
import numpy as np

# functions

def preprocess(text) -> List:
    
    file_content = text
    # breaking into sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokens_list = tokenizer.tokenize(text)
    tokens_list = [s.replace('\n', ' ') for s in tokens_list]
    return tokens_list

def train_test_split_data(text:List, test_size:float=0.1):
    k = int(len(text) * (1 - test_size))
    return text[:k], text[k:]

def vocab_generator(data):
    data = data.translate(str.maketrans('', '', string.punctuation))
    vocabs = data.split()
    return list(vocabs)

def oov_calculator(train, test):
    matches = np.in1d(test,train)
    unseen_tokens = len(matches) - np.count_nonzero(matches)
    oov_rate = unseen_tokens/len(test)
    return oov_rate