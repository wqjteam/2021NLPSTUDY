import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# from sklearn import

# nltk.download('punkt')
"""
读取数据
"""


def read_corpus(corpus_path):
    qlist, alist = [], []
    with open(corpus_path, 'r') as in_file:
        json_corpus = json.load(in_file)['data']
        for article in json_corpus:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    for ans in qa['answers']:
                        qlist.append(qa['question'])
                        alist.append(ans['text'])
    return qlist, alist


"""
理解数据,看一下哪些词比较常见，频率等问题
对于其拉夫定律来说，只有很少的词被经常使用，绝大部分的词很少见
"""


def get_dict(textlist):
    word_dict = defaultdict(lambda: 0)  # 用defaultdict 是为了避免报错
    for text in textlist:
        for token in text.split(" "):
            word_dict[token] += 1
    word_dict = sorted(word_dict.items(), key=lambda item: item[1], reverse=True)
    return dict(word_dict)


def get_topk(n, word_dict):
    res = []
    for word, freq in word_dict.items():
        res.append("{}({})".format(word, freq))
        n = n - 1
        if n == 0:
            return res


# 统计qlist中出现了多少单词
corpus_path = "D:\develop_disk\python/2021NLPSTUDY/input/dialogsystem/train-v2.0.json"
qlist, alist = read_corpus(corpus_path)
q_dict = get_dict(qlist)
word_total_q = sum(q_dict.values())
n_distinctive_word_q = len(q_dict)
# print("There are {} and {} distinctive tokens in the question texts".format(word_total_q, n_distinctive_word_q))
# print(word_total_q)

# 关于词频的统计展示
plt.bar(np.arange(10000), list(q_dict.values())[100:10100])
plt.ylabel("Frequency")
plt.xlabel("Word Order")
plt.title("Word Frequencies of the Question Corpus")
plt.show()

# 在问答显示top10
a_dict = get_dict(alist)
# qTopK = get_topk(10, q_dict)
# aTopK = get_topk(10, a_dict)
# print("问的%s" % (qTopK))
# print("答的%s" % (aTopK))

"""
文本预处理
"""


def isChinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def threadtask(count_thread, thread_index, setk, setstop, list_seq):
    returnlistseq = []
    avge_count = len(list_seq) // count_thread  # 平均每个线程处理的数量, //是地板除法 会舍去小数部分
    lemmatizer = WordNetLemmatizer()  # 词法提取器
    start_time = time.time()
    print("这是第%d个进程开始" % (thread_index))
    # 对数据进行切片
    if thread_index == count_thread:
        list_seq = list_seq[thread_index * avge_count:]
    else:
        list_seq = list_seq[thread_index * avge_count:(thread_index + 1) * avge_count]
    for list_index in range(len(list_seq)):

        list_seq[list_index] = list_seq[list_index].replace('?', '')
        pos_tokens = nltk.pos_tag(nltk.word_tokenize(list_seq[list_index]))
        filtered = []

        for w, pos in pos_tokens:
            if w.lower in setk or w.lower in setstop: continue
            if pos == 'CD':
                filtered.append("#NUM")
            else:
                # if w.find('?') != -1:
                #     w = w.replace('?', '')
                my_pos = get_wordnet_pos(pos)
                if my_pos:
                    # 由于是英文，词太多变，需要转化成原始的词态，在保存
                    # 英语的预处理基本有两种方法 stemming（词干提取，例如规则提取 会有问题） 和 lemmatisation（词性还原）
                    filtered.append(lemmatizer.lemmatize(w, my_pos))
                else:
                    filtered.append(lemmatizer.lemmatize(w))
        returnlistseq.append(' '.join(filtered).lower().strip())
    print("这是第%d个进程结束 耗时为%s min" % (thread_index, (time.time() - start_time) // 60))
    return returnlistseq


def preProcess(list_seq):
    # possession 是财产的意思，拥有
    # 去除停用词
    # 只有英文才有停用词库

    returnlistseq = []
    # 开启线程为5的线程池
    maxpooltread = 1
    list_seq = list_seq[:100]
    # wordnet.ensure_loaded()
    with ThreadPoolExecutor(max_workers=maxpooltread) as t:
        # 提交任务
        all_task = [t.submit(threadtask, maxpooltread, thread_index, setk, setstop, list_seq) for thread_index in
                    range(maxpooltread)]
        # 等待任务完成
        for future_a in as_completed(list(all_task)):
            data = future_a.result()
            returnlistseq.extend(data)
    # for list_index in range(len(list_seq)):
    #     pos_tokens = nltk.pos_tag(nltk.word_tokenize(list_seq[list_index]))
    #     filtered = []
    #
    #     for w, pos in pos_tokens:
    #         if w.lower in setk or w.lower in setstop: continue
    #         if pos == 'CD':
    #             filtered.append("#NUM")
    #         else:
    #             if w.find('?') != -1:
    #                 w = w.replace('?', '')
    #             my_pos = get_wordnet_pos(pos)
    #             if my_pos:
    #                 # 由于是英文，词太多变，需要转化成原始的词态，在保存
    #                 # 英语的预处理基本有两种方法 stemming（词干提取，例如规则提取 会有问题） 和 lemmatisation（词性还原）
    #                 filtered.append(lemmatizer.lemmatize(w, my_pos))
    #             else:
    #                 filtered.append(lemmatizer.lemmatize(w))
    #
    #     returnlistseq.append(' '.join(filtered).lower())
    print("预处理结束")
    return returnlistseq


# 提取词性
def get_wordnet_pos(treebank_tag):
    """
        Convert treebank pos to wordnet pos
    :param self:
    :param treebank_tag:
    :return:
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


"""
对于用户的输入问题，找到相似度最高的TOP5问题，并把5个潜在的答案做返回
"""


def top5results(input_q, tfidfvectorizer):
    """
       给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
       1. 对于用户的输入 input_q 首先做一系列的预处理，然后再转换成tf-idf向量（利用上面的vectorizer)
       2. 计算跟每个库里的问题之间的相似度
       3. 找出相似度最高的top5问题的答案
       """
    # fit 是把所有的数据存入，相当于用数据训练模型
    # transform 将 输入数据 转换成向量（tfidf向量就是array）
    # fit_transform 即训练模型，又输出在模型下面的数据  向量
    q_vec = tfidfvectorizer.transform(preProcess([input_q]))

    top_idxs = []  # top_idxs存放相似度最高的（存在qlist里的）问题的下表
    # hint: 利用priority queue来找出top results. 思考为什么可以这么做？
    # for i in range(q_vec.shape[0]):
    #     similarity=1-np.cos(vectorizer[i,:].todense(),q_vec)
    #     similarity=0 if np.isnan(similarity) else similarity
    num = vectorizer.todense() * q_vec.T  # 向量点乘
    denom = np.reshape(np.linalg.norm(q_vec.todense()) * np.linalg.norm(vectorizer.todense(), axis=1),
                       (100, 1))  # 求模长的乘积
    res = 1 - num / denom
    res[np.isneginf(res)] = 0

    return alist[top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案


setk = set(get_topk(10, dict(sorted(q_dict.items(), key=lambda item: item[1], reverse=False))))
setstop = set(stopwords.words('english'))

qlist = preProcess(qlist)
# alist = preProcess(alist, False)
# 再次处理成array_str


"""
文本表示
"""
tfidfvectorizer = TfidfVectorizer()
vectorizer = tfidfvectorizer.fit_transform(qlist)
# 使用parse martix的方法来表示 内存占会小很多

# 计算稀疏度
sparsity = np.divide(np.prod(vectorizer.shape) - len(vectorizer.nonzero()), np.prod(vectorizer.shape))
print(sparsity)

# TODO: 编写几个测试用例，并输出结果
print(top5results("Who is the president of the United States?", tfidfvectorizer))
print(top5results("What is Shanghai famous for?", tfidfvectorizer))

"""
2.6 利用倒排表的优化
"""
