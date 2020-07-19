import torch

def conv2d(X,K):
    """:param K为卷积核，X为输入
    """
    h,w=K.shape
    Y=torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1)) #卷积核输出维度的计算公式
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=torch.sum(X[i:i+h,j:j+h]*K)
    return Y

