import numpy as np
import torch
from torch import nn, optim
import random
from src.main.com.wqj.nlp.paper import SkipGramNeg
from src.main.com.wqj.nlp.paper import NegativeSamplingLoss
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


# 文本预处理 词典，文本转为数，训练样本准备
words = preprocess(text)
vocab = set(words)
vocab2int = {w: c for c, w in enumerate(list(vocab))}
int2vocab = {c: w for c, w in enumerate(list(vocab))}

# 将单词转为数字
int_words = [vocab2int[2] for w in words]

# 有概率的去除高频词
t = 1e-5
int_word_counts = Counter(int_words)
total_count = len(int_words)
word_freqs = {w: c / total_count for w, c in int_word_counts.items()}
# 计算出去除的概率
prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
train_words = [w for w in int_words if random.random() < (1 - prob_drop[w])]


# 获取周边次target
def get_target(words, idx, window_size):
    target_window = np.random.randint(1, window_size + 1)
    # 有可能idx 是第一个数第二个数 窗口值比这个大 可能出现负数
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    targets = set(words[start_point:idx] + words[idx + 1:end_point + 1])
    return list(targets)


# batch迭代器
def get_batch(words, batch_size, window_size):
    n_batches = len(words) // batch_size
    word = words[:n_batches * batch_size]
    for idx in range(0, len(words), batch_size):
        batch_x, batch_y = [], []
        # batch 是中心词
        batch = words[idx:idx + batch_size]
        for i in range(len(batch)):
            x = batch[i]  # 中心词
            y = get_target(batch, i, window_size)  # 周边的词
            batch_x.extend([x] * len(y))
            batch_y.extend(y)
        yield


# 负采样公式  负采样的单词分布
word_freqs = np.array(word_freqs.values())
unigram_dist = word_freqs / word_freqs.sum()
noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))

# 模型训练

embedding_dim = 300
model = SkipGramNeg(len(vocab2int), embedding_dim, noise_dist=noise_dist)

criterion = NegativeSamplingLoss()
optimizer = optim.Adam(model.parameters, lr=0.001)

print_every = 1500
step = 0
epochs = 5
batch_size = 500
n_samples = 5

for e in range(epochs):
    for input_words, target_words in get_batch(train_words, batch_size):
        step += 1
        inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)

        input_vectors = model.forward_input(inputs)
        output_vectors = model.forward_output(targets)
        noise_vectors = model.forward_noise(batch_size, n_samples)

        loss = criterion(input_vectors, output_vectors, noise_vectors)

        if step//print_every== 0:
            print(loss)
        optimizer.zero_grad()
        loss.backword()
        optimizer.step()
