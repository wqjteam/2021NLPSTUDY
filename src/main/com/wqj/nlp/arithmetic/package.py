import numpy as np

# 物品

commodityWeight = [0, 2, 3, 4, 4, 1]
commodityValue = [0, 2, 6, 4, 5, 3]
stateMartix = np.zeros((6, 11))

for i in range(1, len(commodityWeight)):
    for j in range(1, 11):
        stateMartix[i][j] = stateMartix[i - 1][j]
        if (j >= commodityWeight[i]):
            if ((stateMartix[i - 1][j - commodityWeight[i]] + commodityValue[i]) > stateMartix[i - 1][j]):
                stateMartix[i][j] = stateMartix[i - 1][j - commodityWeight[i]] + commodityValue[i]

print(stateMartix)
