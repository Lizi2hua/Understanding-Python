import numpy as np

arr=np.arange(6*6*3)
arr=arr.reshape(-1,6,3)
print(arr)
print('====')

print(arr[:,0:2])
print('=========')
print(arr[:,:,0:2])
