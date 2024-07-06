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
x=iris.data # 选取前两列作为特征集
y=iris.target

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=300)

# 创建决策树分类器并使用训练集的两个特征进行训练
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
    
# 准确率
print ("traing data Accuracy:",clf.score(x_train, y_train))
print ("testing data Accuracy:",clf.score(x_test, y_test))