import numpy as np


arr=np.arange(16).reshape(4,4)
print(arr[[1,2]])

# print("---------------------")
# print(arr[[1,2],[0,1]])



print("---------------------")
print(arr[[1,2]][:,[1,0]])
