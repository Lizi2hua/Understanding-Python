import numpy as np
import matplotlib.pyplot as plt
x = np.arange(1, 100, 2)
y = np.cos(x)
plt.plot(x, y)
plt.show()
print(np.__version__)