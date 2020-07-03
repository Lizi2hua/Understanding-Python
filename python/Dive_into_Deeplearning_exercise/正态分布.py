import matplotlib.pyplot as plt
import  math
import numpy as np

def normal(z,mu,sigma):
    p=1/math.sqrt(2*math.pi*sigma**2)
    p=p*np.exp(-(0.5/sigma**2)*(z-mu)**2)
    return p

x=np.arange(-7,7,0.01)
para=[(0,1),(0,2),(3,1)]

for mu,sigma in para:
    p=normal(x,mu,sigma)
    plt.plot(x,p)
plt.show()