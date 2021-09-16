import numpy as np

investMartix = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 20, 50, 65, 80, 85, 85],
    [0, 20, 40, 50, 55, 60, 65],
    [0, 25, 60, 85, 100, 110, 115],
    [0, 25, 40, 50, 60, 65, 70]]

stateMartix = np.zeros((5, 7))

statepermutationMartix = [
    [set(tuple()), set(tuple()), set(tuple()), set(tuple()), set(tuple()), set(tuple()), set(tuple())],
    [set(tuple()), set(tuple()), set(tuple()), set(tuple()), set(tuple()), set(tuple()), set(tuple())],
    [set(tuple()), set(tuple()), set(tuple()), set(tuple()), set(tuple()), set(tuple()), set(tuple())],
    [set(tuple()), set(tuple()), set(tuple()), set(tuple()), set(tuple()), set(tuple()), set(tuple())],
    [set(tuple()), set(tuple()), set(tuple()), set(tuple()), set(tuple()), set(tuple()), set(tuple())]]

statepermutationMartix2 = []
for iter in range(5 * 7):
    statepermutationMartix2.append(set(tuple()))

statepermutationMartix2 = np.array(statepermutationMartix2).reshape((5, 7))


def splitInteger(inputdigit):
    tupleset = set()
    for i in range(inputdigit):
        tupleset.add((i, inputdigit - i))
    return tupleset


# 投入工厂
for i in range(1, stateMartix.shape[0]):
    # 投入资金
    for j in range(len(investMartix[0])):
        maxmoney = 0
        for y in range(j + 1):  # j-y是当前工厂投资额的
            if ((stateMartix[i - 1][y] + investMartix[i][j - y]) >= maxmoney):
                # 考虑到可能存在多种情况
                if ((stateMartix[i - 1][y] + investMartix[i][j - y]) > maxmoney):
                    maxmoney = stateMartix[i - 1][y] + investMartix[i][j - y]
                    # statepermutationMartix[i][j] = set(tuple())
                    statepermutationMartix[i][j].clear()  # 存储的时候 （上一层自循环的投资额，当前工厂投资额）
                    statepermutationMartix[i][j].add((y, j - y))

                    stateMartix[i][j] = maxmoney
                else:
                    maxmoney = stateMartix[i - 1][y] + investMartix[i][j - y]
                    statepermutationMartix[i][j].add((y, j - y))

print("状态组合矩阵")
print(statepermutationMartix)
# print(np.array(statepermutationMartix))
print("状态矩阵")
print(stateMartix)

