import os
from nltk.corpus import reuters


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("2021NLPSTUDY\\") + len("2021NLPSTUDY\\")]
# 词典库
vocab = set([line.rstrip() for line in open(rootPath+'input/dialogsystem/vocab.txt')])




# 需要生成所有候选集合
def generate_candidates(word):
    """
    word: 给定的输入（错误的输入）
    返回所有(valid)候选集合
    """
    # 生成编辑距离为1的单词
    # 1.insert 2. delete 3. replace
    # appl: replace: bppl, cppl, aapl, abpl...
    #       insert: bappl, cappl, abppl, acppl....
    #       delete: ppl, apl, app

    # 假设使用26个字符
    letters = 'abcdefghijklmnopqrstuvwxyz'

    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    # insert操作
    inserts = [L + c + R for L, R in splits for c in letters]
    # delete
    deletes = [L + R[1:] for L, R in splits if R]
    # replace
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]

    candidates = set(inserts + deletes + replaces)

    # 过来掉不存在于词典库里面的单词
    return [word for word in candidates if word in vocab]
# 读取语料库
categories = reuters.categories()
corpus = reuters.sents(categories=categories)

# 构建语言模型: bigram
term_count = {}
bigram_count = {}
for doc in corpus:
    doc = ['<s>'] + doc
    for i in range(0, len(doc) - 1):
        # bigram: [i,i+1]
        term = doc[i]
        bigram = doc[i:i + 2]

        if term in term_count:
            term_count[term] += 1
        else:
            term_count[term] = 1
        bigram = ' '.join(bigram)
        if bigram in bigram_count:
            bigram_count[bigram] += 1
        else:
            bigram_count[bigram] = 1

print(term_count)
# sklearn里面有现成的包