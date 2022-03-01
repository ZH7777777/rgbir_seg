import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

n = 8
x = np.array([0.7,31,185,79,100,18.74,19.99,40.58]) # 随机产生10个0~2之间的x坐标
y = np.array([39.7,48.0,51.7,53.8,54.5,55.7,56.4,57.2])
fig = plt.figure(1)
# colors = ['r', 'g', 'y', 'b', 'r', 'c', 'g', 'b', 'k', 'm']
colors = ['k', 'g', 'y', 'm', 'r', 'c', 'g', 'b']
# area = 20*np.arange(1,n+1)
area = np.array([40.0,40.0,40.0,40.0,40.0,40.0,40.0,40.0])
widths = np.array([1,1,1,1,1,1,1,1])
plt.scatter(x, y, s=area, c=colors, linewidths=widths, alpha=0.5, marker='o')
#设置X轴标签
# plt.xlabel('parameter')
#设置Y轴标签
# plt.ylabel('Miou')
# plt.title('test绘图函数')
# 设置横轴的上下限值
plt.xlim(0, 200)
# 设置纵轴的上下限值
plt.ylim(35, 60)
# 设置横轴精准刻度
# plt.xticks(np.arange(np.min(x)-0.2, np.max(x)+0.2, step=0.3))
# 设置纵轴精准刻度
# plt.yticks(np.arange(np.min(y)-0.2, np.max(y)+0.2, step=0.3))
# 设置横轴精准刻度
plt.xticks(np.arange(0, 200, step=20))
# 设置纵轴精准刻度
plt.yticks(np.arange(35, 60, step=5))
#plt.annotate("(" + str(round(x[2],2)) +", "+ str(round(y[2],2)) +")", xy=(x[2], y[2]), fontsize=10, xycoords='data')  #或者
# plt.annotate("({0},{1})".format(round(x[2],2), round(y[2],2)), xy=(x[2], y[2]), fontsize=10, xycoords='data')
# xycoords='data' 以data值为基准
# 设置字体大小为 10
# plt.text(round(x[0],5), round(y[0],5), "MFNet", fontdict={'size': 8, 'color': 'Black'})  # fontdict设置文本字体
# plt.text(round(x[1],5), round(y[1],5), "PSTNet", fontdict={'size': 8, 'color': 'Black'})
# plt.text(round(x[2],5), round(y[2],5), "RTFNet", fontdict={'size': 8, 'color': 'Black'})
# plt.text(round(x[3],5), round(y[3],5), "FuNNet", fontdict={'size': 8, 'color': 'Black'})
# plt.text(round(x[4],5), round(y[4],5), "FuseSeg", fontdict={'size': 8, 'color': 'Black'})
# plt.text(round(x[5],5), round(y[5],5), "MFFENet(s)", fontdict={'size': 8, 'color': 'Black'})
# plt.text(round(x[6],5), round(y[6],5), "MFFFENet(M)", fontdict={'size': 8, 'color': 'Black'})
# plt.text(round(x[7],5), round(y[7],5), "Ours", fontdict={'size': 8, 'color': 'Black'})

# Add text to the axes.
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.savefig('plot.svg', dpi=10000, bbox_inches='tight', transparent=False)
# plt.legend(['绘图测试'], loc=2, fontsize = 10)
# plt.legend(['绘图测试'], loc='upper left', markerscale = 0.5, fontsize = 10) #这个也可
#  markerscale：The relative size of legend markers compared with the originally drawn ones.
plt.show()
