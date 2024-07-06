import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
import time

# 导入数据
wine_quality = fetch_ucirepo(id=186) 
X = wine_quality.data.features
y = wine_quality.data.targets

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# 计算损失函数
def huber_loss(y, y_pred, delta):
    diff = y - y_pred
    abs_diff = np.abs(diff)
    quadratic_part = np.minimum(abs_diff, delta)
    linear_part = abs_diff - quadratic_part
    loss = 0.5 * quadratic_part**2 + delta * linear_part
    return loss

# MBSGD算法
def mbsgd(X, y, learning_rate, num_epochs, batch_size, delta, threshold):
    num_samples, num_features = X.shape
    num_batches = num_samples // batch_size
    
    # print()

    weights = np.zeros(num_features)
    bias = 0.0

    # 存储每次迭代的损失值
    loss_history = []
    time_history = []

    total_time = 0.0

    for epoch in range(num_epochs):
        # 记录当前时间
        start_time = time.time()
        
        # 打乱训练数据
        if batch_size!=X_train.shape[0]:   
            permutation = np.random.permutation(num_samples)
            X = X[permutation]
            y = y[permutation]
        
        # 按批次进行训练
        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size        

            # 提取当前批次的数据
            X_batch = X[start:end]
            y_batch = y[start:end]
            
            y_pred = np.dot(X_batch, weights) + bias
            y_pred = y_pred.reshape(-1,1)            
            
            # 计算梯度
            diff = y_batch - y_pred
            gradient = -np.dot(X_batch.T, diff) / batch_size
            gradient = gradient.reshape(-1)
            bias_gradient = -np.mean(diff)

            # 更新模型参数
            weights -= learning_rate * gradient
            bias -= learning_rate * bias_gradient

        # 计算当前损失
        y_pred = np.dot(X, weights) + bias
        loss = huber_loss(y, y_pred, delta)
        average_loss = np.mean(loss)

        # 记录总时间
        end_time = time.time()
        total_time += end_time - start_time
        time_history.append(total_time)

        # 存储损失值
        loss_history.append(average_loss)

        # 打印当前迭代的损失
        # print(f"Epoch {epoch+1}/{num_epochs} - Loss: {average_loss} - Total Time: {total_time:.4f} seconds")

        # 检查损失是否低于阈值，如果是则终止迭代
        if average_loss < threshold:
            print(f"Loss is below threshold ({threshold}). Stopping training.")
            break

    return weights, bias, loss_history, time_history

# 使用训练好的模型进行预测，通过残差平方和进行评估
def calculate_rss(X, y, weights, bias):
    y_pred = np.dot(X, weights) + bias    
    residuals = y - y_pred   
    rss = np.sum(residuals ** 2)   
    return rss

learning_rate = 0.00001
num_epochs = 100
delta = 1.0
threshold = 0.1

start = 1
end = 2000
num_points = 10

# 计算步长
step = (end - start) / (num_points - 1)
# 生成等间隔的整数序列
numbers = np.arange(start, end+1, (end-start)//(num_points-1))
 
# 绘制三种算法损失函数随时间的变化图
def plot_loss_over_time(X_train, y_train, learning_rate, num_epochs, delta, threshold):
    batch_sizes = [1, 32 ,X_train.shape[0]]
    #batch_size=1表示SGD，=32表示MBSGD，=X_teain.shape[0]表示GD
    plt.figure(figsize=(10, 6))   
    for batch_size in batch_sizes:
        weights, bias, loss_history, time_history = mbsgd(X_train, y_train, learning_rate, num_epochs, batch_size, delta, threshold)
        rss = calculate_rss(X_test, y_test, weights, bias)
        print(f"Residual Sum of Squares (RSS): {rss}")
        plt.plot(time_history, loss_history, label=f'Batch Size = {batch_size}')

    plt.xlabel('Total Time (seconds)')
    plt.ylabel('Loss')
    plt.title('Loss vs. Total Time for Different Batch Sizes')
    plt.legend()
    plt.xscale('log')
    plt.show()
    
# 绘制残差平方和和batch_sizes的关系图
def plot_rss_over_batch_size(X_train, y_train, learning_rate, num_epochs, delta, threshold):
    # 选择其中的 20 个整数
    batch_sizes = numbers[:num_points].tolist()
    rss_values = []
    for batch_size in batch_sizes:
        weights, bias, loss_history, time_history = mbsgd(X_train, y_train, learning_rate, num_epochs, batch_size, delta, threshold)
        rss = calculate_rss(X_test, y_test, weights, bias)
        rss_values.append(rss)
    plt.plot(batch_sizes, rss_values)
    plt.xlabel('Batch Size')
    plt.ylabel('RSS')
    plt.title('RSS vs. Batch Size')
    plt.grid(True)
    plt.show()
    
plot_loss_over_time(X_train, y_train, learning_rate, num_epochs, delta, threshold)
plot_rss_over_batch_size(X_train, y_train, learning_rate, 20, delta, threshold)