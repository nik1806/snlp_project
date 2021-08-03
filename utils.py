# dependencies
import string
from typing import List, Dict
import nltk.data

# functions

def preprocess(text) -> List:
    
    # file_content = text.lower()
    # file_content = file_content.translate(str.maketrans('', '', string.punctuation))
    file_content = text
    # breaking into sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokens_list = tokenizer.tokenize(text)
    # tokens_list = file_content.split('\n')
    # tokens_list = [s.strip('\n') for s in tokens_list]
    tokens_list = [s.replace('\n', ' ') for s in tokens_list]
    return tokens_list

def train_test_split_data(text:List, test_size:float=0.1):
    k = int(len(text) * (1 - test_size)) # may be randomize the split later
    return text[:k], text[k:]