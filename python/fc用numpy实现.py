# two layer fc net
import numpy as np
import matplotlib.pyplot as plt
import time


# H is hidden dimension
N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# randomly inintialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)


lr = 1e-6
start_time=time.time()
for e in range(500):
    # forward pass
    h = x.dot(w1)
    #shape=(64 1000)*(1000 100)=(64 100)
    h_relu = np.maximum(h, 0)

    y_pred = h_relu.dot(w2)
    #(64 10)

    # compute loss
    loss = np.square(y - y_pred)
    # print("epoch:",e,"loss:",loss)

    # backprop:compute the gradient(!!!!!!)
    grad_y_pred=2.0*(y_pred-y)
    grad_w2=h_relu.T.dot(grad_y_pred)
    grad_h_relu=grad_y_pred.dot(w2.T)
    grad_h=grad_h_relu.copy()
    grad_h[h<0]=0
    grad_w1=x.T.dot(grad_h)

    #update weights
    w1-=lr*grad_w1
    w2-=lr*grad_w2
end_time=time.time()
time=end_time-start_time
print("using %f second"%time)


