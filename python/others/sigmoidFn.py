import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))

data=np.random.randn(1,1024)
print(data)
for i in data:
    x=i
    b=2
    w=3
    z=w*x+b
    res=sigmoid(z)
    plt.plot(x,res,'.')
    plt.show()