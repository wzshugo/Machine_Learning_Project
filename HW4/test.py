import torch

class CifarDataset(torch.utils.data.Dataset):
    def __init__ (self, root_dir):
    # 初始化数据集，包含一系列图片与对应的标签
        super().__init__()
        raise NotImplementedError
    def __len__ (self):
    # 返回数据集的大小
        raise NotImplementedError
    def __getitem__ (self, index):
    # 返回该数据集的第index个数据样本
        raise NotImplementedError

train_dataset = CifarDataset(TRAIN_DIRECTORY_PATH)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)