import os
import numpy as np

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("2021NLPSTUDY\\") + len("2021NLPSTUDY\\")]
# 词典库


tag2id, id2tag = {}, {}
word2id, id2word = {}, {}

for line in open(rootPath + 'input/dialogsystem/traindata.txt'):
    items = line.split('/')
    word, tag = items[0], items[1].rstrip()

    if word not in word2id:
        word2id[word] = len(word2id)
        id2word[len(id2word)] = word
    if tag not in tag2id:
        tag2id[tag] = len(tag2id)
        id2tag[len(id2tag)] = tag

M = len(word2id)
N = len(tag2id)
print(M)
print(N)

pai = np.zeros(N)  # 每个词性出现在句子中第一个位置还来
A = np.zeros((N, M))  # A[i][j] 给定tag i，出现单词j的概率，N
B = np.zeros((N, N))  # B[i][j] 出现tagi  再出现tagj的概率

prev_tag = ' '
for line in open(rootPath + 'input/dialogsystem/traindata.txt'):
    items = line.split('/')
    wordId, tagId = word2id[items[0]], tag2id[items[1].rstrip()]
    if prev_tag == "":  # 这意味着是句子的开始
        pai[tagId] += 1
        A[tagId][wordId] += 1
    else:  # 不是句子的开始
        A[tagId][wordId] += 1
        B[tag2id[prev_tag]][tagId] += 1
    if items[0] == '.':  # 结束
        prev_tag = ""
    else:
        prev_tag = items[1].rstrip()

# 把数字变成概率
pai = pai / sum(pai)
for i in range(N):
    A[i] /= sum(A[i])
    B[i] /= sum(B[i])


def log(v):
    if (v == 0):
        return np.log(v + 0.0000000001)
    else:
        return np.log(v)


def viterbi(x, pai, A, B):
    """
    :param x:  用户输入字符串
    :param pai: 初始单词的概率
    :param A: 给定的Tag，每个单词出现的概率
    :param B: tag之间转移的概率
    :return:
    """

    x = [word2id[word] for word in x.split(" ")]
    T = len(X)
    dp = np.zeros((T, N))  # dp[i][j]:w1...wj,假设wi的tag是低j个tag
    ptr = np.array([[0 for x in range(N)] for y in range(T)])  # T*N 整数类型矩阵
    for j in range(N):  # basecase for DP 算法
        dp[0][j] = log(pai[j]) + log(A[j][x[0]])

    for i in range(1, T):  # 每个单词
        for j in range(N):  # 每个词性
            dp[i][j] = -9999
            for k in range(N):  # 从每一个K可以到达j
                score = dp[i - 1][k] + log(B[k][j]) + log[A[j][x[i]]]
                if score > dp[i][j]:
                    dp[i][j] = score
                    ptr[i][j] = k

    # 把最好的sequence打印出来
    best_seq = [0] * T  # best_seq=[1,5,3,5...,16]
    # step1：找出对饮最后一个单词的词性
    best_seq[T - 1] = np.argmax(dp[T - 1])
    # step2： 从后到前求出单词词性
    for i in range(T - 2, -1, -1):
        best_seq[i] = ptr[i + 1][best_seq[i + 1]]
    # 到目前为止 best_seq存放了对应x的词性序列
    for i in range(len(best_seq)):
        print(id2tag[best_seq[i]])
