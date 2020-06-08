import torch

x = torch.Tensor([[2, 1], [4, 3], [5, 2]])
print(x)
y = torch.Tensor([[4],[9],[7] ])
xt = x.t()
print(xt)
x_tx = xt@x
x_txinvers=torch.inverse(x_tx)

w=x_txinvers@xt@y
print(w)
# w=(xt*x)^-1*xt*y
