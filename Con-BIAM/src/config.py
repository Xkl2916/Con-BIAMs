import os
import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn
# http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/language/
# path to a pretrained word embedding file
word_emb_path = 'D:\python\PyCharm 2022.2.1\projects\BBFN\home\yingting\Glove\glove.840B.300d.txt'      # 这个是已经是已经使用Bert预处理完的词向量编码
# word_emb_path = '/home/yingting/Glove/glove.840B.300d.txt'
# 这里增加了一个断言， 就是说 如果word_emb_path 确实存在， 就继续往下运行， 如果不存在程序就在这里停止报错
assert(word_emb_path is not None)

username = Path.home().name                                             # 用户名 Xkl
project_dir = Path(__file__).resolve().parent.parent
print(project_dir)
print(username)
# D:\git project\CMU-MultimodalSDK
sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')                     # 这里是提前下载好的工具包，可以方便的处理多模态数据
data_dir = project_dir.joinpath('dataset')
# print(data_dir)  D:\python\PyCharm 2022.2.1\projects\BBFN\dataset
# data_dir = project_dir.joinpath('MOSI')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
    'MOSEI'), 'ur_funny': data_dir.joinpath('UR_FUNNY')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}

def str2bool(v):                                                        # 这个函数暂时没有弄明白是干什么的
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, data, mode='train'):
        # 这里的数据集，mode 如果不传参数， 就是默认为训练集， 如果mode传进来了参数， 则按照传进来的为主
        """Configuration Class: set kwargs as class attributes with setattr"""
        # if kwargs is not None:
        #     for key, value in kwargs.items():
        #         if key == 'optimizer':
        #             value = optimizer_dict[value]
        #         if key == 'activation':
        #             value = activation_dict[value]
        #         setattr(self, key, value)

        # Dataset directory: ex) ./datasets/cornell/
        self.dataset_dir = data_dict[data.lower()]
        print("*********************")
        print(self.dataset_dir)
        print("*********************")
        self.sdk_dir = sdk_dir
        self.mode = mode                                            # 训练集 or 验证集 or 测试集

        # Glove path
        self.word_emb_path = word_emb_path                          # 词向量编码

        # Data Split ex) 'train', 'valid', 'test'
        # self.data_dir = self.dataset_dir.joinpath(self.mode)
        self.data_dir = self.dataset_dir                            # 数据路径

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str

# 原来的batch_size = 32， 就是如果不传进来参数就设置32， 传进来了就不设置32
def get_config(dataset='mosi', mode='train', batch_size=32, use_bert=False):

    config = Config(data=dataset, mode=mode)                        # 创建一个Config类对象

    config.dataset = dataset                                        # 数据集
    config.batch_size = batch_size                                  # 批量大小
    config.use_bert = use_bert                                      # 是否使用Bert， 这里因为已经预处理出Glove了， 所以use_bert=False

    # if dataset == "mosi":
    #     config.num_classes = 1                                    # 是正向情绪还是负向情绪 [-3 - 3]
    # elif dataset == "mosei":
    #     config.num_classes = 1
    # elif dataset == "ur_funny":
    #     config.num_classes = 2
    # else:
    #     print("No dataset mentioned")
    #     exit()

    # Namespace => Dictionary
    # kwargs = vars(kwargs)
    # kwargs.update(optional_kwargs)

    return config
