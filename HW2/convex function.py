import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def f(x, y, a, b):
    z = 0
    for i in range(N):
        z += np.square(1 / (1 + np.exp(-(x + y * a[i]))) - b[i])
    return z

# 生成数据
N = 100  # 参数个数
x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# 生成随机参数
a = np.random.randn(N)
b = np.random.randn(N)

# 计算函数的值
Z = f(X, Y, a, b)

# 创建画布
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三维图形
ax.plot_surface(X, Y, Z, cmap='viridis')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()