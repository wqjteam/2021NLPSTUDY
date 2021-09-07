import numpy as np
import torch
from torch import nn, optim
import random
from collections import Counter


#损失函数
class NegativeSamplingLoss(nn.Module):
    # 因为集成了矩阵运算，所以要继承nn.Module
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        batch_size, embed_size = input_vectors.shape

        input_vectors = input_vectors.view(batch_size, embed_size, 1)

        # 把矩阵反过来，才能进行运算，不然维书不一样
        output_vectors = output_vectors.view(batch_size, 1, embed_size)

        # 进行点击运算
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()

        out_loss = out_loss.squeeze() #降维 成了batch——size*1

        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()

        noise_loss = noise_loss.squeeze().sum(1)


        return -(out_loss + noise_loss).mean()
