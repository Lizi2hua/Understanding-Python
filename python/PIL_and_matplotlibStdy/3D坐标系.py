from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
z = np.random.normal(0, 1, 100)

# 创建画板
fig = plt.figure()
# 创建3D坐标系
ax = Axes3D(fig)
'''for ... 实时画图'''
ax.scatter(x, y, z, c='red', marker='x',label='girl')
plt.legend()
'''...实时画图'''
plt.show()


# # 实时画图
ax = []
ay = []
az=[]
fig2=plt.figure()
ax2=Axes3D(fig2)
plt.ion()

for i in range(100):
    ax.append(20 * np.cos(i))
    ay.append(20*np.sin(i))
    az.append(i*3.5)

    #clf清除画板内容，cla清除画板
    plt.cla
    ax2.scatter(ax,ay,az,c='b',marker='v',s=10,label='rain')
    # s:size
    # plt.legend()
    # 动态图里面显示图例也是动态的
    # 显示图例
    plt.pause(0.01)

plt.ioff()
plt.show()
# ax1=[]
# ay1=[]
#
# plt.ion()
# for k in range(1000):
#     ax1.append(20*np.cos(k))
#     ay1.append(20*np.sin(k))
#
#     plt.clf()
#     plt.scatter(ax1,ay1,c='r',marker='v')
#     plt.pause(0.001)
# plt.ioff()
# plt.show()

