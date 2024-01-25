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
batch_size = 16
lr = 1e-6

# 设置device
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载model，preprocess
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)


# 测试数据加载器
class MyTestDataset(Dataset):
    def __init__(self, img_folder, category_file, preprocess):
        self.img_folder = img_folder
        self.category_file = category_file
        self.preprocess = preprocess

        # 读取分类标签文件
        with open(self.category_file, 'r') as f:
            lines = f.readlines()
            self.categories = {line.split(':')[0]: line.split(':')[1].strip() for line in lines}

        # 获取图路径列表
        self.imgs = [os.path.join(self.img_folder, img) for img in os.listdir(self.img_folder) if
                     img.endswith('.jpg')]
        # 生成分类标签列表
        self.categories_list = [f"{self.categories[os.path.basename(img)]}" for img in self.imgs]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        category = self.categories_list[idx]

        # 加载图片
        image = Image.open(img_path)
        # 图片预处理
        image_tensor = self.preprocess(image)

        # 返回图片特征向量和分类标签
        return image_tensor, category



# 加载model, optimizer
def load_model_components(file_path, device):
    with open(file_path, 'rb') as file:
        components = pickle.load(file)

    # 初始化model和optimizer
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.load_state_dict(components['model'])

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    optimizer.load_state_dict(components['optimizer'])

    return model, optimizer


# 测试函数
def test():
    load_model, load_optimizer = load_model_components('/content/drive/MyDrive/my_clip_finetune/models/model_components1.pth',
                                                       device=device)


    # 读取测试数据集文件和类别文件路径
    test_img_folder_path = '/content/drive/MyDrive/my_clip_finetune/dataset/my_CIFAR10/my_test_cifar10'
    test_category_file_path = '/content/drive/MyDrive/my_clip_finetune/dataset/my_CIFAR10/my_test_cifar10_category.txt'
    # 使用os.path.basename获取路径中的最后一个目录名
    directory_name = os.path.basename(test_img_folder_path)

    # 创建测试数据集实例
    test_dataset = MyTestDataset(test_img_folder_path, test_category_file_path, preprocess)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 定义目标类别列表
    infos = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 对目标类别进行token处理
    infos_tensor = clip.tokenize(infos).to(device)

    correct_all = 0  # 记录全部测试集正确预测的数量

    with torch.no_grad():
      # 迭代测试数据
      for batch_num, (images_tensor, targets_category) in enumerate(test_dataloader):
          correct_batch = 0  # 记录当前batch正确预测的数量
          # 将图片tensor转移到设备上
          images_tensor = images_tensor.to(device)

          # 使用加载的模型进行预测
          logits_per_image, _ = load_model(images_tensor, infos_tensor)

          # 对每个样本的预测结果进行softmax
          probs = logits_per_image.softmax(dim=-1)

          # 取最大概率对应的类别作为预测类别
          predicted_category_idx = torch.argmax(probs, dim=-1)
          # 将索引映射到infos中的具体类别
          predicted_category = [infos[idx] for idx in predicted_category_idx]

          # 将预测类别与真实类别进行比较
          correct_batch += sum(1 for pred, target in zip(predicted_category, targets_category) if pred == target)

          # 获取当前图片的信息
          '''print('-' * 5, f"image:{image_name},targets_category:{targets_category}", '-' * 5)
          print(f"infos:{infos}")
          print(f"probs:{probs.cpu().numpy()}")
          print(f"predicted_category:{predicted_category}")'''


          print(
              f"batch_num:{batch_num + 1} accurary on this batch:{(correct_batch / float(len(targets_category))) * 100:.2f}%")

          correct_all += correct_batch  # 将当前batch正缺数加到全部数据正确数上

      print('-' * 5,
            f"Accurary on the {os.path.basename(test_img_folder_path)}:{(correct_all / float(len(test_dataset))) * 100:.2f}%",
            '-' * 5)


if __name__ == "__main__":
  test()  # 测试
