import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

class CifarDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
    # 初始化数据集，包含一系列图片与对应的标签
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.images = []  # 存放图像路径
        self.labels = []  # 存放对应的标签

        # 遍历数据集文件夹，获取图像路径和标签
        classes = sorted(os.listdir(self.root_dir))
        for i, class_name in enumerate(classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    self.images.append(image_path)
                    self.labels.append(i)

    def __len__(self):
    # 返回数据集的大小
        return len(self.images)

    def __getitem__(self, index):
    # 返回该数据集的第index个数据样本
        image_path = self.images[index]
        label = self.labels[index]

        # 读取图像
        image = Image.open(image_path)

        # 可选的图像预处理
        if self.transform is not None:
            image = self.transform(image)

        return image, label

# 定义数据集存放的路径
TRAIN_DIRECTORY_PATH = "D://下载//cifar10_train"
TEST_DIRECTORY_PATH = "D://下载//cifar10_test"

# 定义数据预处理的方式
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 创建训练集数据集对象和数据加载器
train_dataset = CifarDataset(TRAIN_DIRECTORY_PATH, transform=transform)
train_len = train_dataset.__len__()
print(train_len)
train0 = train_dataset.__getitem__(0)
print(train0)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 创建测试集数据集对象和数据加载器
test_dataset = CifarDataset(TEST_DIRECTORY_PATH, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 计算每份中的样本数量
total_train_samples = len(train_dataset)
samples_train = total_train_samples // 10
total_test_samples = len(test_dataset)
samples_test = total_test_samples // 10

# 创建存储训练集和测试集的列表
train_indices = []
test_indices = []

# 按顺序平均分成十份，并从每份中选择一定比例的样本作为训练集和测试集
for i in range(10):
    start_train_index = i * samples_train
    end_train_index = (i + 1) * samples_train
    
    start_test_index = i * samples_test
    end_test_index = (i + 1) * samples_test

    # 从当前份的样本中选择一定比例的索引作为训练集
    train_end_index = start_train_index + int(0.25 * samples_train)
    train_indices.extend(list(range(start_train_index, train_end_index)))

    # 从当前份的样本中选择一定比例的索引作为测试集
    test_end_index = start_test_index + int(1 * samples_test)
    test_indices.extend(list(range(start_test_index, test_end_index)))

# 创建训练集和测试集的 Subset
train_subset = torch.utils.data.Subset(train_dataset, train_indices)
test_subset = torch.utils.data.Subset(test_dataset, test_indices)

# 定义全连接神经网络模型
class FCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate, num_hidden_layers):
        super(FCNModel, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layers = nn.ModuleList()

        self.hidden_layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i in range(self.num_hidden_layers):
            x = self.hidden_layers[i](x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

# 定义卷积神经网络模型
class CNNModel(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc = nn.Linear(8 * 8 * 32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train(model, dataloader, optimizer, criterion, learning_rate, num_epochs = 10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

def calculate_accuracy(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

# 定义超参数
batch_sizes = [32, 64, 128, 256]  # 不同的batch size
hidden_sizes = [100, 200, 300, 400]  # 不同的隐藏层中神经元个数
num_layers = [1, 2, 3, 4]  # 不同的隐藏层数量
dropout_rates = [0.0, 0.1, 0.2, 0.3]  # 不同的 dropout rate 值
learning_rates = [0.0001, 0.0005, 0.001, 0.002]  # 不同的 learning rate 值

# 创建训练集和测试集的数据加载器
train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_sizes[0], shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_subset, batch_size=batch_sizes[0], shuffle=False)

fcn_accuracies = []  # 存储 FCNModel 的准确性
cnn_accuracies = []  # 存储 CNNModel 的准确性
              
def hyper_parameter(batch_size = 64, hidden_size = 100, dropout_rate = 0.1, learning_rate = 0.001, num_layer = 1):
    # 创建模型实例
    input_size = 32 * 32 * 3
    num_classes = 10
    fcn_model = FCNModel(input_size, hidden_size, num_classes, dropout_rate, num_layer)
    cnn_model = CNNModel(num_classes, dropout_rate)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fcn_model.parameters())
    optimizer = optim.Adam(cnn_model.parameters())

    print(f"Batch Size: {batch_size}, Hidden Size: {hidden_size}, Dropout Rate: {dropout_rate}, Learning Rate: {learning_rate}, Num Layers: {num_layer}")
    print("Experimenting with FCNModel:")
    train(fcn_model, train_dataloader, optimizer, criterion, learning_rate)
    print("Experimenting with CNNModel:")
    train(cnn_model, train_dataloader, optimizer, criterion, learning_rate)
        
    fcn_model_accuracy = calculate_accuracy(fcn_model, test_dataloader)
    print("FCNModel Accuracy:", fcn_model_accuracy)
    cnn_model_accuracy = calculate_accuracy(cnn_model, test_dataloader)
    print("CNNModel Accuracy:", cnn_model_accuracy)

    fcn_accuracies.append(fcn_model_accuracy)
    cnn_accuracies.append(cnn_model_accuracy)
    
for batch_size in batch_sizes:
    hyper_parameter(batch_size=batch_size)

fig, axes = plt.subplots(5, 1, figsize=(8, 12))

# 绘制准确性曲线
axes[0].plot(batch_sizes, fcn_accuracies, label='FCNModel')
axes[0].plot(batch_sizes, cnn_accuracies, label='CNNModel')
axes[0].set_xlabel('Batch Size')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy vs Batch Size')
axes[0].legend() 

fcn_accuracies = []
cnn_accuracies = []   
for hidden_size in hidden_sizes:
    hyper_parameter(hidden_size=hidden_size)
    
# 绘制准确性曲线
axes[1].plot(hidden_sizes, fcn_accuracies, label='FCNModel')
axes[1].plot(hidden_sizes, cnn_accuracies, label='CNNModel')
axes[1].set_xlabel('Hidden Size')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy vs Hidden Size')
axes[1].legend()

fcn_accuracies = []
cnn_accuracies = []      
for num_layer in num_layers:
    hyper_parameter(num_layer=num_layer)
    
# 绘制准确性曲线
axes[2].plot(num_layers, fcn_accuracies, label='FCNModel')
axes[2].plot(num_layers, cnn_accuracies, label='CNNModel')
axes[2].set_xlabel('Num Layers')
axes[2].set_ylabel('Accuracy')
axes[2].set_title('Accuracy vs Num Layers')
axes[2].legend() 

fcn_accuracies = []
cnn_accuracies = []      
for dropout_rate in dropout_rates:
    hyper_parameter(dropout_rate=dropout_rate)
    
# 绘制准确性曲线
axes[3].plot(dropout_rates, fcn_accuracies, label='FCNModel')
axes[3].plot(dropout_rates, cnn_accuracies, label='CNNModel')
axes[3].set_xlabel('Dropout Rate')
axes[3].set_ylabel('Accuracy')
axes[3].set_title('Accuracy vs Dropout Rate')
axes[3].legend()

fcn_accuracies = []
cnn_accuracies = []  
for learning_rate in learning_rates:
    hyper_parameter(learning_rate=learning_rate)

# 绘制准确性曲线
axes[4].plot(learning_rates, fcn_accuracies, label='FCNModel')
axes[4].plot(learning_rates, cnn_accuracies, label='CNNModel')
axes[4].set_xlabel('Learning Rate')
axes[4].set_ylabel('Accuracy')
axes[4].set_title('Accuracy vs Learning Rate')
axes[4].legend()

# 调整子图之间的间距
plt.tight_layout()

# 显示图像
plt.show()