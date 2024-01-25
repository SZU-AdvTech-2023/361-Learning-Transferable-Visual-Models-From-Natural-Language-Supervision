# 重要说明！！！

    所有脚本文件中使用的路径都是在Colab上的绝对路径，
    如果需要下载到本地使用，请修改相应位置的路径

# 代码运行平台

    Colab pro版

# 环境要求

    根据requirements.txt文件中提示的包使用Colab指令下载必要的包
    
    配置clip时请使用以下指令：
    !pip install git+https://github.com/openai/CLIP.git

# 数据集准备

    在dataset文件夹中，有以下两个文件夹：
    CIFAR10：存放cifar10原来的数据集，使用data_decompression.py解压缩
    
    my_CIFAR10：存放本项目使用的数据集，使用data_prepare.py制作

# 训练和测试分别包含在两个文件夹中

    训练文件夹：包含对照组和实验组两个训练文件，两组训练后参数分别保存在models文件夹中
    对照组train_control_group：在原分辨率数据集上训练CLIP模型得到model_components1.pth
    实验组train_experimental_group：在超分数据集机上训练CLIP模型得到model_components2.pth


    测试文件夹：包含对照组与实验组在测试集上的测试文件
    对照组test_control_group：使用model_components1.pth训练参数在测试集上测试
    实验组test_experimental_group：使用model_components2.pth训练参数在测试集上测试
# 训练

运行以下代码执行对照组：

```
%cd train
!python train_control_group.py 
```

运行以下代码执行实验组：

```
%cd train
!python train_experimental_group.py 
```



# 测试

注：必须运行过训练代码后才能执行测试代码，即本仓库并不提供预训练模型！！！

运行以下代码执行对照组：

```
%cd test
!python test_control_group.py 
```

运行以下代码执行实验组：

```
%cd train
!python test_experimental_group.py 
```

