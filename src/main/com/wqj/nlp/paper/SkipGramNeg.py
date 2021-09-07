import numpy as np
import torch
from torch import nn, optim
import random
from collections import Counter


class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist=None):
        super().__init__()
        self.n_embed = n_embed
        self.n_vocab = n_vocab
        self.noise_dist = noise_dist

        # 给embedding层定义输入输入和输出
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)

        # 初始化权重
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    # 把输入的中心进过嵌入层，变成向量
    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors

    # 把输出的中心进过嵌入层，变成词嵌入向量
    def forward_output(self, output_words):
        output_vectors = self.in_embed(output_words)
        return output_vectors

    def forward_noise(self, batch_size, n_samples):
        if self.noise_dist is None: #如果没有初始化的噪声，所有单词进行等概率的采样
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist
        noise_words = torch.multinomial(noise_dist,batch_size*n_samples,replacement=True)
        noise_dist= self.out_embed(noise_words).view(batch_size,n_samples,self.n_embed)
        return output_vectors
