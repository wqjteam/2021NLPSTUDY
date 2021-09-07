import numpy as np
import torch
from torch import nn, optim
import random
from collections import Counter

# 读取文件，先去掉低频词
file_path = ""
with open(file_path) as f:
    # 如果计算机性能比较高，可以选择性的不去除那些低频次
    text = f.read()


def preprocess(text, freq):
    text = text.lower()
    text = text.replace(".", "<PERIO>")  # 处理一些特殊符号 。。。可能会有很多
    words = text.split()
    word_counts = Counter(words)
    ternmed_words = [word for word in words if word_counts[word] > freq]
    return ternmed_words
