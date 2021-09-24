import numpy as np


# 先验条件：
# 2、  已知情况
# 隐含的身体状况={健康，发烧}
# 可观察的感觉状态={正常、冷、头晕}
# 月儿预判的阿鲁的身体状态的概率分布={健康：0.6，发烧：0.4}
# 月儿认为的阿鲁的身体健康的转换概率分布={
# 健康->健康：0.7
# 健康->发烧：0.3
# 发烧->健康：0.4
# 发烧->发烧：0.6
# }
# 月儿认为的在相应的健康状况条件下，阿鲁的感觉的概率分布={
# 健康，正常：0.5，冷：0.4，头晕：0.1；
# 发烧，正常：0.1，冷：0.3，头晕：0.6
# }
# 已知条件
# 阿鲁连续三天的身体感觉依次是：正常、冷、头晕。


# P(今天健康) = P(正常|健康)*P(健康|初始情况) = 0.5 * 0.6 = 0.3

# P(今天发烧) = P(正常|发烧)*P(发烧|初始情况) = 0.1 * 0.4 = 0.04


# 那么第二天有四种情况，由于第一天的发烧或者健康转换到第二天的发烧或者健康。

# P(前一天发烧，今天发烧) = P(前一天发烧) * P(发烧->发烧)*P(冷 | 发烧) = 0.04 * 0.6 * 0.3 = 0.0072

# P(前一天发烧，今天健康) = P(前一天发烧) * P(发烧->健康)*P(冷 | 健康) = 0.04 * 0.4 * 0.4 = 0.0064

# P(前一天健康，今天健康) = P(前一天健康) * P(健康->健康)*P(冷 | 健康) = 0.3 * 0.7 * 0.4 = 0.084

# P(前一天健康，今天发烧) = P(前一天健康) * P(健康->发烧)*P(冷 | 发烧) = 0.3 * 0.3 * .03 = 0.027

# 那么可以认为，第二天最可能的状态是：健康。


# HMM
# 的三大问题
# （1）评估问题(概率计算问题)
# 即给定观测序列 O=O1,O2,O3…Ot和模型参数λ=(A,B,pi)，怎样有效计算这一观测序列出现的概率.
# (Forward-backward算法)
# （2）解码问题(预测问题)
# 即给定观测序列 O=O1,O2,O3…Ot和模型参数λ=(A,B,pi)，怎样寻找满足这种观察序列意义上最优的隐含状态序列S。
# (viterbi算法,近似算法)
# （3）学习问题
# 即HMM的模型参数λ=(A,B,pi)未知，如何求出这3个参数以使观测序列O=O1,O2,O3…Ot的概率尽可能的大.
# (极大似然估计的方法估计参数,Baum-Welch,EM算法)

def viterbi(trainsion_probility, emission_probility, pi, obs_seq):
    # trainsion_probility A矩阵 转移概率  行隐藏状态A      转移到列隐藏B的概率
    # emission_probility  B矩阵 发射概率   行隐藏状态A      显示出列A1 的概率
    # pi 隐藏状态分布， 所有的隐藏状态加起来等于1
    row = trainsion_probility.shape[0]
    col = obs_seq.shape[0]
    StateMartix = np.zeros(
        (row, col))  # 是一个三行三列的矩阵  第一列 代表第一天状态的可能性 StateMartix[0,0]代表第一天健康的状态可能性，StateMartix[1,0]代表第一天不健康健康的状态可能性

    # 用来存放上一层的路径
    detelist = list()
    for i in range(row * col):
        detelist.append(set())
    statepermutationMartix = np.array(detelist).reshape((row, col))

    # 初始状态
    StateMartix[:, 0] = np.max(pi * np.transpose(emission_probility[:, obs_seq[0]]),
                               0)  # 就是 观测值 *  A矩阵的隐藏状态发生显性状态的概率转置（显性状态由某个隐性状态转来的概率）
    # 再存储来源的隐形矩阵 要存t-1下标，记录来源
    statepermutationMartix[:0] = list
    for t in range(1, col):  # t表示第几列
        list_max = []
        for n in range(row):  # 计算现在隐藏状态转为其他(一个一个的算)的隐藏状态的概率 是一行一行从上向下计算
            # 在隐形状态中 要取隐形状态概率大的 乘出来是一个矩阵 取每列的最大值，存储起来
            Ft = StateMartix[:, t - 1] * trainsion_probility[:, n]  # 上一次的所有可能状态 * B矩阵的隐藏状态转为显性状态的转置（显性状态由某个隐性状态转来的概率）
            maxP = np.max(Ft)
            list_max.append(maxP)
            indexs = np.where(Ft == maxP)
            for index in indexs:
                statepermutationMartix[n, t].add((index[0], t - 1))  # t-1表示前一列
        # 将计算得来的最大的概率存起来
        StateMartix[:, t] = np.array(list_max) * np.transpose(emission_probility[:, obs_seq[t]])
        # 存储历史路径，和寻找每一列的max （一列的max代表 隐形状态可能最大的比例）

    return StateMartix, statepermutationMartix


if __name__ == '__main__':
    # 隐马尔可夫模型λ=(A, B, pai)
    # pai是初始状态概率分布，初始状态个数=np.shape(pai)[0]
    # 在所有数据中 来自各个盒子的占比 ，所有的想加起来等于1
    pai = np.array([[0.2], [0.4], [0.4]])

    # A是状态转移概率分布，状态集合Q的大小N=np.shape(A)[0]
    # 同理，统计所有样本中，在前面是盒子1在变成盒子2的次数，再除以盒子1出现的总次数，便得到由盒子1转移到盒子2的概率分布，其他可得
    # 从下给定A可知：Q={盒1, 盒2, 盒3}, N=3
    A = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])

    # B是观测概率分布，观测集合V的大小T=np.shape(B)[1]
    # 统计训练数据中，状态为j并观测为k的频数，除以训练数据中状态j出现的次数，其他同理可得
    # 从下面给定的B可知：V={红，白}，T=2
    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])

    # 观测序列
    Observe = np.array([[0],
                        [1],
                        [0]])  # 0表示红色，1表示白，就是(红，白，红)观测序列

    # 通过Observe 观测序列 预测隐藏序列
    StateMartix, statepermutationMartix = viterbi(A, B, pai, Observe)
    print(StateMartix)
    print(statepermutationMartix)
