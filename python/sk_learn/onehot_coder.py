import numpy as np
def onehot_encode(target,cls):
    list=np.zeros((1,cls-1),dtype=int)
    return np.insert(list,target,1)

onehot=onehot_encode(89,100)
print(onehot)

for i in range (len(onehot)):
    if onehot[i]==1:
        target=i
        break
print(target)