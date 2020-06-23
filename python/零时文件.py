import numpy as np
arr=np.array([[[1,1,1]]])
arr_1=np.pad(arr,(1,1),constant_values=(0,2))
print(arr_1)