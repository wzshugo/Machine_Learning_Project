from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 转换格式
def pgm_to_vector(image_path):
    # 读取PGM图像文件
    with Image.open(image_path) as img:
        # 转换为灰度图像
        grayscale_image = img.convert('L')
        # 将灰度图像转换为NumPy数组
        image_array = np.array(grayscale_image)
        # 将图像展平为一维向量
        image_vector = image_array.flatten()
        return image_vector

# 读取文件
def folder_to_array(folder_path):
    image_vectors = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pgm"):
            image_path = os.path.join(folder_path, filename)
            image_vector = pgm_to_vector(image_path)
            image_vectors.append(image_vector)
    image_array = np.array(image_vectors)
    return image_array

# 显示图片
def vector_to_image(image_vector, image_shape):
    # 将向量重新调整为图像形状
    image_array = image_vector.reshape(image_shape)
    return image_array

# 如果维度<100，扩展数组
def expand_array(arr):
    if len(arr) < 100:
        diff = 100 - len(arr)
        padding = np.zeros(diff)
        arr = np.concatenate((arr, padding))
    return arr

# 导入训练数据
folder_path_0 = "D://下载//作业3-手写字的数据集//train//train0"
folder_path_1 = "D://下载//作业3-手写字的数据集//train//train1"
folder_path_2 = "D://下载//作业3-手写字的数据集//train//train2"

# 导入测试数据
folder_path_test = "D://下载//作业3-手写字的数据集//test//test_total"

# 将训练数据转化为5000x784的数组
zero_train = folder_to_array(folder_path_0)
one_train = folder_to_array(folder_path_1)
two_train = folder_to_array(folder_path_2)
y_train = np.concatenate((np.zeros(5000), np.ones(5000), np.full(5000, 2)))

# 将测试数据转化为1500x784的数组
x_test = folder_to_array(folder_path_test)
y_test = array = np.concatenate((np.zeros(500), np.ones(500), np.full(500, 2)))

# 合并训练数据为15000x784的数组
total_train = np.concatenate((zero_train, one_train, two_train), axis=0)

# pca
n_components = int(input("请输入降维后的维数："))

# 分别对三个训练集进行操作
train_datas = [total_train, zero_train, two_train]
train_datas_pca = []

for train_data in train_datas:
    # 显示平均图
    average_train = np.mean(train_data, axis=0)
    image = vector_to_image(average_train, (28,28))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
    
    pca = PCA(n_components = n_components, svd_solver = 'full').fit(train_data)
    train_datas_pca.append(pca.transform(train_data))
    
    # 获取特征值
    eigenvalues = pca.explained_variance_

    # 获取特征向量
    eigenvectors = pca.components_

    # 取前100个特征值的列表，按照从大到小排序
    eigenvalues = expand_array(eigenvalues[:100])
    
    # 绘制特征值大小和标号的图像
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, 101), eigenvalues, marker='o', linestyle='-', color='b')
    plt.xlabel('eigenvector indexes')
    plt.ylabel('eigenvector values')
    plt.title('The first 20 eigenvectors change with index')
    plt.grid(True)
    plt.show()

    # 创建一个4行5列的子图网格
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10, 8))

    # 绘制前20个特征向量（作为图像）
    for i in range(20):
        image = vector_to_image(eigenvectors[i], (28, 28))
        row = i // 5
        col = i % 5
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f"eigenvector({i+1})")

    plt.tight_layout()
    plt.show()
  
pca = PCA(n_components = n_components, svd_solver = 'full').fit(x_test)
x_test_pca = pca.transform(x_test)

k_values = range(1, 101)  # 设置K值范围
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_datas_pca[0], y_train)

    # 预测测试集
    y_pred = knn.predict(x_test_pca)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# 绘制曲线图
plt.plot(k_values, accuracies)
plt.xlabel('the value of K')
plt.ylabel('accuracy')
plt.title('accuracies change with K')
plt.grid(True)
plt.show()