import torch
import argparse
import numpy as np

from utils import *
from torch.utils.data import DataLoader
from solver import Solver
from config import get_config
from data_loader import get_loader
#  ***********************************************************************************************************
#
#  *************       http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/language/      **************************
#
#  ***********************************************************************************************************


# argparse模块的作用是用于解析命令行参数
parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
# 使用的模型的名字
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')

# Tasks 这个是什么东西? 三个模态不应该是 v, a, t 吗？ 这里的l代表的什么意思？？？
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
# 是否使用模态对齐的实验
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
# 使用的数据集，论文使用了三个数据集，默认是mosi, mosei数据集比mosi大很多
parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi','mosei','ur_funny'],
                    help='dataset to use (default: mosei)')
# 存储数据集的路径
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts 一些函数中的丢弃率, 为了是防止模型过拟合
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,               # 残差块 的丢弃率
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,               # 输出层的丢弃率
                    help='output layer dropout')
parser.add_argument('--div_dropout', type=float, default=0.1)               #

# Embedding 是否使用BERT来对文本进行编码，得到文本的词向量
parser.add_argument('--use_bert', action='store_true', help='whether to use bert \
                    to encode text inputs (default: False)')

# Losses 损失 这个损失是在模态融合的过程中，每一层融合完之后都需要一个特征分离器，然后一直加到最后一层， 最后在和总的分类任务的损失相加。
parser.add_argument('--lambda_d', type=float, default=0.1, help='portion of discriminator loss added to total loss (default: 0.1)')

# Architecture 模型结构 这是就是一个transformer的一个架构
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')                # 有几层编码器(encoder)，默认是5 比原来的transformer少了3层
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')    # 用几个头， 默认用5个， 原来transformer是用8个
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')          # padding mask 长度不一样需要补零， 后面要消除0的影响
parser.add_argument('--attn_hidden_size', type=int, default=40,
                    help='The size of hiddens in all transformer blocks')               # 这个就是那个全连接层隐藏层的大小, 原来应该是2048？
parser.add_argument('--uni_nlevels', type=int, default=3,
                    help='number of transformer blocks for unimodal attention')
parser.add_argument('--enc_layers', type=int, default=1,                                # 文章是用的双向GRU， 来进行捕捉上下文之间的联系, 用一层就可以了
                    help='Layers of GRU or LSTM in sequence encoder')
parser.add_argument('--use_disc', action='store_true',                                  # 这个应该是有的，就是为了保持模态之间的独立性，增加一个特征分离器，会有损失
                    help='whether to add a discriminator to the domain-invariant encoder and the corresponding loss to the final training process')

parser.add_argument('--proj_type', type=str, default='cnn',help='network type for input projection', choices=['LINEAR', 'CNN','LSTM','GRU'])
parser.add_argument('--lksize', type=int, default=3,                                    # 3 x 3 的卷积核大小
                    help='Kernel size of language projection CNN')
parser.add_argument('--vksize', type=int, default=3,                                    # 3 x 3 的卷积核大小
                    help='Kernel size of visual projection CNN')
parser.add_argument('--aksize', type=int, default=3,                                    # 3 x 3 的卷积核大小
                    help='Kernel size of accoustic projection CNN')

# Tuning
parser.add_argument('--batch_size', type=int, default=1, metavar='N',                  # 批量大小 原来是24， 我改成1了
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,                                  # 这个应该是防止梯度爆炸的吧？？？
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=5e-4,                                   # 学习率 0.0001 ，有点低了吧？？？
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',                                # 优化器， 常用Adam
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=1,                               # 他这里默认是训练40轮， 我这里改成了只训练了1轮
                    help='number of epochs (default: 10)')
parser.add_argument('--when', type=int, default=20,                                     # 为了防止学习率过大，在收敛到全局最优点的时候会来回摆荡， 每20轮降低一次
                    help='when to decay learning rate (default: 20)')                   # 学习初期的可以大一些， 到后面快收敛的时候调小
parser.add_argument('--batch_chunk', type=int, default=1,                               # 这是啥？ 没太弄明白？？？
                    help='number of chunks per batch (default: 1)')


# Logistics
parser.add_argument('--log_interval', type=int, default=30,                             # 结果记录的频率或者间隔
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,                                   # 随机种子
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',                                   # 为啥不用cuda? 没GPU
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')

#Control experiment
##################################################################################################
#Control training method
parser.add_argument('--train_method', type=str, default='missing',                      # 是否增加噪声
                        help='one of {missing, g_noise}, missing means set to zero noise, g_noise means set to Gaussian Noise')
#Control the modality of change during training                                         # 训练时改变模态
parser.add_argument('--train_changed_modal', type=str, default='language', help='one of {language, video, audio}')
#Control the percentage of change during training
parser.add_argument('--train_changed_pct', type=float, default=0, help='Control the percentage of change during training')


#Control testing method
parser.add_argument('--test_method', type=str, default='missing',  
                        help='one of {missing, g_noise}, missing means set to zero noise, g_noise means set to Gaussian Noise')
#Control the modality of change during testing
parser.add_argument('--test_changed_modal', type=str, default='language', help='one of {language, video, audio}')
#Control the percentage of change during training
parser.add_argument('--test_changed_pct', type=float, default=0, help='Control the percentage of change during testing')

#Distinguish between eval and test                                                       # 区分是测试还是评估
parser.add_argument('--is_test', action='store_true', help='Distinguish between eval and test')
#######################################################################################################

args = parser.parse_args()

# set numpy manual seed
# np.random.seed(args.seed)
args.seed = 1000

# 设置 CPU 生成随机数的 种子 ，方便下次复现实验结果 |||||  不设随机种子，生成随机数, 设置随机种子，使得每次运行代码生成的随机数都一样
torch.manual_seed(args.seed)

# 这个是啥？？？ 有效的部分
valid_partial_mode = args.lonly + args.vonly + args.aonly                                 # 这个打印出来是 0 ， 0 = args.lonly = args.vonly = args.aonly


# configurations for data_loader
dataset = str.lower(args.dataset.strip())

batch_size = args.batch_size
print(batch_size)

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

# use_cuda = False
# 这里我自己修改成True了
use_cuda = True

output_dim_dict = {                                             # 最后结果的输出维度
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8,
    # ir_funny 预测说话人是否表达了幽默
    # 'ur_funny': 1   # comment this if using BCELoss           # BCELoss 二分类交叉熵损失
    'ur_funny': 2 # comment this if using CrossEntropyLoss
}

criterion_dict = {                                  # 正则化函数, mosi 用 L1损失函数
    'mosi': 'L1Loss',
    'iemocap': 'CrossEntropyLoss',
    'ur_funny': 'CrossEntropyLoss'
}

torch.set_default_tensor_type('torch.FloatTensor')   #origin        # 设置tensor的数据类型， 一般为浮点数
#torch.set_default_tensor_type('torch.cuda.FloatTensor') #added        # 有GPU设置这个，没有用上面的

# 这个是来判断是否使用GPU加速
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        # print("我来使用了") 结果是可以进来
        torch.cuda.manual_seed_all(args.seed)
        # torch.set_default_tensor_type('torch.cuda.FloatTensor') #origin

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True

####################################################################
#######       Load the dataset (aligned or non-aligned)       ######
####################################################################

print("Start loading the data....")
# 训练集
train_config = get_config(dataset, mode='train', batch_size=args.batch_size, use_bert=args.use_bert)
# print(train_config)
"""
{'batch_size': 1,
 'data_dir': WindowsPath('D:/python/PyCharm 2022.2.1/projects/BBFN/dataset/MOSI'),
 'dataset': 'mosi',
 'dataset_dir': WindowsPath('D:/python/PyCharm 2022.2.1/projects/BBFN/dataset/MOSI'),
 'mode': 'train',
 'sdk_dir': WindowsPath('D:/python/PyCharm 2022.2.1/projects/BBFN/CMU-MultimodalSDK'),
 'use_bert': False,
 'word_emb_path': 'D:\\python\\PyCharm '
                  '2022.2.1\\projects\\BBFN\\home\\yingting\\Glove\\glove.840B.300d.txt'}
"""
# 验证集 用于在训练过程中检验模型的状态，收敛情况。验证集通常用于调整超参数， 监控模型是否发生过拟合
valid_config = get_config(dataset, mode='valid', batch_size=args.batch_size, use_bert=args.use_bert)
# 测试集， 这个是最后的测试集
test_config = get_config(dataset, mode='test',  batch_size=args.batch_size, use_bert=args.use_bert)

# print(train_config)

hyp_params = args
# Namespace(f='', model='MulT', vonly=True, aonly=True, lonly=True, aligned=False,
# dataset='mosi', data_path='data', attn_dropout=0.1, attn_dropout_a=0.0, attn_dropout_v=0.0, relu_dropout=0.1,
# embed_dropout=0.25, res_dropout=0.1, out_dropout=0.0, div_dropout=0.1, use_bert=False, lambda_d=0.1, nlevels=5,
# num_heads=5, attn_mask=True, attn_hidden_size=40, uni_nlevels=3, enc_layers=1, use_disc=False, proj_type='cnn',
# lksize=3, vksize=3, aksize=3, batch_size=24, clip=0.8, lr=0.0005, optim='Adam', num_epochs=10, when=20,
# batch_chunk=1, log_interval=30, seed=1000, no_cuda=False, name='mult', train_method='missing',
# train_changed_modal='language', train_changed_pct=0, test_method='missing', test_changed_modal='language',
# test_changed_pct=0, is_test=False)

# pretrained_emb saved in train_config here
train_loader = get_loader(hyp_params, train_config, shuffle=True)               # 训练集随机打乱一下 shuffle = True
valid_loader = get_loader(hyp_params, valid_config, shuffle=False)
test_loader = get_loader(hyp_params, test_config, shuffle=False)

print('Finish loading the data....')
if not args.aligned:                                                            # 看是否是加载的对齐的数据集
    print("### Note: You are running in unaligned mode.")
####################################################################
#
# Hyperparameters  超参数
#
####################################################################

# # 正向传播时：开启自动求导的异常侦测
torch.autograd.set_detect_anomaly(True)

# addintional appending
# print(train_config.word2id)
# print("****************************************")
# print(train_config.pretrained_emb)
hyp_params.word2id = train_config.word2id                       # 每一个单词的编号， 比如现在用的文本一共有2728个独一无二的此， 对这些词进行编号
hyp_params.pretrained_emb = train_config.pretrained_emb         # 每一个词的词向量都是300维的， 一共2728个词
# print(hyp_params.pretrained_emb.size())                       # torch.Size([2729, 300])
# exit()


# architecture parameters
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_config.lav_dim
# The resulting vector lengths for the three datasets
# (MOSI, MOSEI and UR-FUNNY) are 47, 35 and 75 respectively
print(hyp_params.orig_d_l)
print(hyp_params.orig_d_a)
print(hyp_params.orig_d_v)
exit()
# print(train_config.lav_dim)                                   # 根据论文中提到的，  三个数据集(MOSI, MOSEI, ur-funcy)的文本维度分别为 (47, 35, 75)
                                                                # 这个地方应该我理解错了， 这里的输出为（300 74 47）， 应该是Bert中隐藏层的大小
                                                                # 具体可以看data_load 中 42行出定义

if hyp_params.use_bert:
    hyp_params.orig_d_l = 768
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_config.lav_len     # 这个长度是什么？？？ #################################
print(train_config.lav_len)
hyp_params.layers = args.nlevels
hyp_params.l_ksize = args.lksize                                                # 卷积核的大小
hyp_params.v_ksize = args.vksize
hyp_params.a_ksize = args.aksize

hyp_params.proj_type = args.proj_type.lower()                                   # 网络类型
hyp_params.num_enc_layers = args.enc_layers

hyp_params.use_cuda = use_cuda
hyp_params.dataset = hyp_params.data = dataset
hyp_params.when = args.when
hyp_params.attn_dim = args.attn_hidden_size                                     # 隐藏层的大小维度
hyp_params.batch_chunk = args.batch_chunk
# hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = train_len, valid_len, test_len
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
# hyp_params.criterion = criterion_dict.get(dataset, 'MAELoss')
hyp_params.criterion = criterion_dict.get(dataset, 'MSELoss')                   # 损失函数类型， 如果没有就是MSELoss


if __name__ == '__main__':
    solver = Solver(hyp_params, train_loader=train_loader, dev_loader=valid_loader, # 实例化一个Net
                    test_loader=test_loader, is_train=True)
    solver.train_and_eval()                                                         # 开始训练和评估
    exit()
# 如果遇到这个报错: program aborting due to control-C event， 请看下方链接
# https://blog.csdn.net/qq_51116518/article/details/120267208