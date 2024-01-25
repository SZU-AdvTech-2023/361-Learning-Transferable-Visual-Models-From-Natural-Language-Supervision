import time

import pickle
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn.functional import softmax
from torch.optim import lr_scheduler

import clip
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# 设置超参数
epoches = 25
batch_size = 32
lr = 1e-6

# 设置device
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载model，preprocess
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)


# 训练数据加载器
class MyTrainDataset(Dataset):
    def __init__(self, img_folder, category_file, preprocess):
        self.img_folder = img_folder
        self.category_file = category_file
        self.preprocess = preprocess

        # 读取分类标签文件
        with open(self.category_file, 'r') as f:
            lines = f.readlines()
            self.categories = {line.split(':')[0]: line.split(':')[1].strip() for line in lines}

        # 获取图像路径列表
        self.imgs = [os.path.join(self.img_folder, img) for img in os.listdir(self.img_folder) if
                     img.endswith('.jpg')]
        # 生成分类标签列表
        self.categories_list = [f"{self.categories[os.path.basename(img)]}" for img in self.imgs]

        # 标签预处理
        self.tokens = clip.tokenize(self.categories_list)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        token = self.tokens[idx]

        # 加载图片
        image = Image.open(img_path)
        # 图像预处理
        image_tensor = self.preprocess(image)

        return image_tensor, token



# 训练数据集存放路径
img_folder_path =  '/content/drive/MyDrive/my_clip_finetune/dataset/my_CIFAR10/my_train_cifar10'
category_file_path = '/content/drive/MyDrive/my_clip_finetune/dataset/my_CIFAR10/my_train_cifar10_category.txt'

# 创建训练数据集实例
my_dataset = MyTrainDataset(img_folder=img_folder_path, category_file=category_file_path, preprocess=preprocess)
my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

# 设置损失，使用交叉熵损失测量两个概率分布之间的差异。
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# 设置优化器，lr：学习率，betas：梯度的移动平均值，eps：防止除0加在分母上，weight_decay：正则化化系数防止过拟合加载损失函数上
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)


# 更新模型参数类型
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


# 保存model，optimizer
def sava_model_components(file_path, model, optimizer):
    components = {
        'models': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    with open(file_path, 'wb') as file:
        pickle.dump(components, file)


# 训练函数
def train():
    loss_list = []  # 记录损失
    start_time = time.time()  # 记录开始时间

    for epoch in range(epoches):
        loss_epoch = 0  # 汇总当前epoch的所有loss
        for batch_num, (images_tensor, tokens) in enumerate(my_dataloader):
            # 将图片和标签token转移到device设备
            images_tensor = images_tensor.to(device)
            tokens = tokens.to(device)

            optimizer.zero_grad()  # 优化器梯度清零

            # 计算 图像-文本相似度、文本-图像相似度
            logits_per_image, logits_per_text = model(images_tensor, tokens)
            # 获得每张图像的ground_truth
            ground_truth = torch.arange(len(images_tensor), dtype=torch.long, device=device)
            # 计算平均损失
            cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth)) / 2

            print(f"epoch:{epoch + 1} batch:{batch_num + 1} loss:{cur_loss}")  # 输出当前batch的损失
            loss_epoch += cur_loss

            # 反向传播更新model参数
            cur_loss.backward()
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

            # 更新lr
            # scheduler.step()

            # 如果当前batch是当前epoch的最后一个batch，计算batch的平均loss加入到loss_loss列表中
            if batch_num == len(my_dataloader) - 1:
                loss_list.append((loss_epoch.cpu().detach().numpy() / float(len(my_dataloader))))

    # 保存model和组件
    sava_model_components('/content/drive/MyDrive/my_clip_finetune/models/model_components1.pth', model, optimizer)

    # 计算训练花费的时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finish training! Training time: {elapsed_time} ")

    # 可视化训练和训练损失
    fig = plt.figure()  # 创建一个画布对象
    plt.plot(range(1, epoches + 1), loss_list, color='blue')  # 绘制训练损失曲线（蓝色）
    plt.xlabel('epochs')  # 添加 x 轴标签
    plt.ylabel('losses')  # 添加 y 轴标签
    plt.show()  # 显示图形


if __name__ == "__main__":
  train()  # 训练

