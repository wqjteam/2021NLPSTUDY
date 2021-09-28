import numpy as np
import json

"""
读取数据
"""


def read_corpus(corpus_path):
    qlist, alist = [], []
    with open(corpus_path, 'r') as in_file:
        json_corpus = json.load(in_file)['data']
    for article in json_corpus:
        for paragraph in article['paragrapgs']:
            for qa in paragraph['pas']:
                for ans in qa['answers']:
                    qlist.append(qa['question'])
                    alist.append(ans['text'])
    return qlist, alist


"""
理解数据
"""


def get_dict():
    pass
