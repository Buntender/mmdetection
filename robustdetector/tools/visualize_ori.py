import numpy as np

'''降维过程'''
X = np.array([[10, 56], [80, 21], [21, 30], [11, 81]]) # 数据(4X3)
labels = np.array([1, 0, 1, 1]) # 每一行数据对应的标签(例如二分类问题)

'''可视化过程'''
from matplotlib import pyplot as plt
scatter = plt.scatter(X[:,0], X[:,1], 20, labels) # labels为每一行对应标签，20为标记大小
legend = plt.legend(scatter.legend_elements(num=1)[0], ["c1", "c2"], loc="upper left", title="Ranking")
plt.savefig("transH.png") #保存图片
plt.show()