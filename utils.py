# dependencies
import string
from typing import List, Dict

# functions

def preprocess(text) -> List:
    file_content = text.lower()
    file_content = file_content.translate(str.maketrans('', '', string.punctuation))
    tokens_list = file_content.split()
    return tokens_list

def train_test_split_data(text:List, test_size:float=0.1):
    k = int(len(text) * (1 - test_size)) # may be randomize the split later
    return text[:k], text[k:]