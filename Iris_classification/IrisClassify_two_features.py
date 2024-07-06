# 导入必要的包
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 数据集准备
iris = load_iris()
x=iris.data[:,:2] # 选取前两列作为特征集
y=iris.target

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=300)

# 创建决策树分类器并使用训练集的两个特征进行训练
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
    
# 准确率
print ("traing data Accuracy:",clf.score(x_train, y_train))
print ("testing data Accuracy:",clf.score(x_test, y_test))

# 测试模型（绘图）
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # 生成网格采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

grid_pred = clf.predict(grid_test)      # 预测分类值
grid_pred = grid_pred.reshape(x1.shape)  # 使之与输入的形状相同

cm_light = ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF']) # 背景颜色映射
cm_dark = ListedColormap(['g', 'r', 'b']) # 数据颜色映射

plt.pcolormesh(x1, x2, grid_pred, cmap=cm_light) # 绘制背景

plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=40, cmap=cm_dark) # 绘制数据
plt.scatter(x_test[:, 0], x_test[:, 1], s=100, facecolor='yellow', zorder=1, marker='+')  # 突出测试集数据

plt.xlabel("Sepal.Length",fontsize=15)
plt.ylabel("Sepal.Width",fontsize=15)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('KNN.iris', fontsize=20)
plt.grid()
plt.show()