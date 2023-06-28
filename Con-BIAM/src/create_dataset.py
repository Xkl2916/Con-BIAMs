import sys
import mmsdk
import os
import re
import pickle
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from mmsdk import mmdatasdk as md
from subprocess import check_call, CalledProcessError
import torch
import torch.nn as nn


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)                                                     # 把得到的数据放到某一个文件中，方便下次调用， 不用再重新获取了
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)                                                   # 加载提前处理好的数据

# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']                                                          # 一些特殊的字符，用这个来表示
PAD = word2id['<pad>']                                                          # 加的padding??

# turn off the word2id - define a named function here to allow for pickling     #
def return_unk():
    return UNK

def load_emb(w2i, path_to_embedding, embedding_size=300, embedding_vocab=2196017, init_emb=None):
    # 如果初始化的emb是空的话，就随机生成len(w2i) 个长度为embedding_size大小的emb_mat
    if init_emb is None:
        emb_mat = np.random.randn(len(w2i), embedding_size)
    else:
        emb_mat = init_emb
    # 打开词向量编码文件
    f = open(path_to_embedding, 'r', encoding='UTF-8')
    found = 0
    for line in tqdm_notebook(f, total=embedding_vocab):
        # 读取每一行数据， 以空格为切片， 转化为列表
        content = line.strip().split()
        # 转为 numpy.ndarray 类型
        vector = np.asarray(list(map(lambda x: float(x), content[-300:])))
        word = ' '.join(content[:-300])
        # 统计 一共有多少个单词
        if word in w2i:
            idx = w2i[word]
            emb_mat[idx, :] = vector
            found += 1
    print(f"Found {found} words in the embedding file.")
    return torch.tensor(emb_mat).float()

# 这里有三个数据集类， 我只看了MOSI， 剩下的两个大致上和MOSI差不多， 没细看
class MOSI:
    def __init__(self, config):

        # 加载处理多模态数据的模型 CMU-MultimodalSDK
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))
        
        DATA_PATH = str(config.dataset_dir)
        print("我的数据路径")
        print(DATA_PATH)
        CACHE_PATH = DATA_PATH + '/bbfn_embedding_and_mapping.pt'

        # 如果说下面的数据已经存在了， 直接加载， 如果不存在就重新创建
        # If cached data if already exists
        # 如果缓存数据已经存在
        try:
            self.train = load_pickle(DATA_PATH + '/bbfn_train.pkl')
            self.dev = load_pickle(DATA_PATH + '/bbfn_dev.pkl')
            self.test = load_pickle(DATA_PATH + '/bbfn_test.pkl')
            self.pretrained_emb, self.word2id = torch.load(CACHE_PATH)

        except:

            # create folders for storing the data
            # 创建用于存储数据的文件夹
            print("我还没进去，我的路径是")
            print(DATA_PATH)
            if not os.path.exists(DATA_PATH): # 首先是判断这个路径是否存在， 如果不存在则
                # print(os.path.exists(DATA_PATH)) False 说明这个不存在
                print(' '.join(['mkdir', '-p', DATA_PATH]))
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

            # download highlevel features, low-level (raw) data and labels for the dataset MOSI
            # if the files are already present, instead of downloading it you just load it yourself.
            # here we use CMU_MOSI dataset as example.
            DATASET = md.cmu_mosi

            # 它下面的这个代码是要下载数据集， 我这里运行的时候会报错， 所以我直接去网站上下载下来了， 就把第二个try的地方注释了

            # try:
            #     md.mmdataset(DATASET.labels, './')
            #     # md.mmdataset(DATASET.highlevel, DATA_PATH)
            # except RuntimeError:
            #     print("High-level features have been downloaded previously.")
            # # mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel, 'cmumosi/')
            #
            # try:
            #
            #     md.mmdataset(DATASET.raw, DATA_PATH)
            # except RuntimeError:
            #     print("Raw data have been downloaded previously.")
            #
            # try:
            #     md.mmdataset(DATASET.labels, DATA_PATH)
            # except RuntimeError:
            #     print("Labels have been downloaded previously.")

            # 这里就是它那个网站上每一个模态下有好几个数据，这里是每一个模态下选择哪个数据进行后面的训练.
            # define your different modalities - refer to the filenames of the CSD files
            # visual_field = 'CMU_MOSI_VisualFacet_4.1'
            visual_field = 'CMU_MOSI_Visual_Facet_41'
            acoustic_field = 'CMU_MOSI_COVAREP'
            text_field = 'CMU_MOSI_TimestampedWords'

            # 三个模态特征， 分别为文本、视觉、音频
            features = [
                text_field, 
                visual_field, 
                acoustic_field
            ]

            # 这个是每一个模态特征数据的路径，弄成一个字典， 方便后面操作
            recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
            # print(recipe)
            # {'CMU_MOSI_TimestampedWords': 'D:\\python\\PyCharm 2022.2.1\\projects\\BBFN\\dataset\\MOSI\\CMU_MOSI_TimestampedWords.csd', 'CMU_MOSI_Visual_Facet_41': 'D:\\python\\PyCharm 2022.2.1\\projects\\BBFN\\dataset\\MOSI\\CMU_MOSI_Visual_Facet_41.csd', 'CMU_MOSI_COVAREP': 'D:\\python\\PyCharm 2022.2.1\\projects\\BBFN\\dataset\\MOSI\\CMU_MOSI_COVAREP.csd'}
            dataset = md.mmdataset(recipe)

            # we define a simple averaging function that does not depend on intervals
            # 我们定义了一个不依赖区间的简单的平均函数
            def avg(intervals: np.array, features: np.array) -> np.array:
                try:
                    return np.average(features, axis=0)
                except:
                    return features

            # first we align to words with averaging, collapse_function receives a list of functions
            # 首先使用平均值函数对齐单词, collapse_function接受函数列表
            dataset.align(text_field, collapse_functions=[avg])
            # 标签
            label_field = 'CMU_MOSI_Opinion_Labels'

            # we add and align to lables to obtain labeled segments
            # this time we don't apply collapse functions so that the temporal sequences are preserved
            # 获得情感标签， 为了保留时间序列， 不使用collapse functions
            label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
            dataset.add_computational_sequences(label_recipe, destination=None)
            dataset.align(label_field)

            # 通过视频id 获得训练集、 验证集、 测试集
            # obtain the train/dev/test splits - these splits are based on video IDs
            train_split = DATASET.standard_folds.standard_train_fold
            dev_split = DATASET.standard_folds.standard_valid_fold
            test_split = DATASET.standard_folds.standard_test_fold


            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            # 这个是干啥的？？？
            EPS = 1e-6

            

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            # 这是用了一个正则表达式
            # 就是用来抽取视频ID的
            pattern = re.compile('(.*)\[.*\]')
            # 有一些模态数据可能在处理之后 序列没有对齐， 对于这些数据，我们应丢弃， 所以这个计数器就是统计丢弃了多少的数据点
            num_drop = 0 # a counter to count how many data points went into some processing issues

            for segment in dataset[label_field].keys():
                # 在已经对齐的数据集中获得视频标签和特征
                # get the video ID and the features out of the aligned dataset
                # 视频标号
                vid = re.search(pattern, segment).group(1)
                # 情感标签
                label = dataset[label_field][segment]['features']
                # 文本
                _words = dataset[text_field][segment]['features']
                # 视频特征
                _visual = dataset[visual_field][segment]['features']
                # 音频特征
                _acoustic = dataset[acoustic_field][segment]['features']

                # if the sequences are not same length after alignment, there must be some problem with some modalities
                # we should drop it or inspect the data again
                # 序列长度不一样，就丢弃
                if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
                    print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
                    num_drop += 1
                    continue

                # remove nan values 去掉一些NAN值， 上一步的操作只能去掉序列不一样长的数据， 不能去掉长度都为0的数据
                label = np.nan_to_num(label)
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)


                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                # 这个难道是是，对于语音部分中间有一些停顿的地方，对于这些地方，我们应该去掉这一部分，随之对应的视频和文本方面也应该去掉？  是这个意思吗？
                actual_words = []       # 实际单词  看来有可能是我上面理解的
                words = []
                visual = []
                acoustic = []
                for i, word in enumerate(_words):
                    if word[0] != b'sp':
                        actual_words.append(word[0].decode('utf-8'))
                        words.append(word2id[word[0].decode('utf-8')]) # SDK stores strings as bytes, decode into strings here
                        visual.append(_visual[i, :])
                        acoustic.append(_acoustic[i, :])

                # 最终的文本、视频、语音特征
                words = np.asarray(words)
                visual = np.asarray(visual)
                acoustic = np.asarray(acoustic)


                # z-normalization per instance and remove nan/infs
                # 对特征进行归一化处理
                visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
                acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))
                # 就是根据视频标号，将对应的特征划分到相应的训练集 or 验证集 or 测试集中
                if vid in train_split:
                    train.append(((words, visual, acoustic, actual_words), label, segment))
                elif vid in dev_split:
                    dev.append(((words, visual, acoustic, actual_words), label, segment))
                elif vid in test_split:
                    test.append(((words, visual, acoustic, actual_words), label, segment))
                else:
                    print(f"Found video that doesn't belong to any splits: {vid}")

            print(f"Total number of {num_drop} datapoints have been dropped.")

            word2id.default_factory = return_unk

            # 存储 glove 编码的cache 路径
            # Save glove embeddings cache too
            self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            torch.save((pretrained_emb, word2id), CACHE_PATH)

            # Save pickles
            to_pickle(train, DATA_PATH + '/bbfn_train.pkl')
            to_pickle(dev, DATA_PATH + '/bbfn_dev.pkl')
            to_pickle(test, DATA_PATH + '/bbfn_test.pkl')

    # 获得数据集
    def get_data(self, mode):
        # 返回三个值， 第一个是数据、 第二个是文本单词的编号， 这个编号是所有独一无二的单词编号 以及提前训练好的word编码
        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "valid":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()




class MOSEI:
    def __init__(self, config):

        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))
        
        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/bbfn_embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/bbfn_train.pkl')
            self.dev = load_pickle(DATA_PATH + '/bbfn_dev.pkl')
            self.test = load_pickle(DATA_PATH + '/bbfn_test.pkl')
            self.pretrained_emb, self.word2id = torch.load(CACHE_PATH)

        except:

            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)


            # download highlevel features, low-level (raw) data and labels for the dataset MOSEI
            # if the files are already present, instead of downloading it you just load it yourself.
            DATASET = md.cmu_mosei
            try:
                md.mmdataset(DATASET.highlevel, DATA_PATH)
            except RuntimeError:
                print("High-level features have been downloaded previously.")

            try:
                md.mmdataset(DATASET.raw, DATA_PATH)
            except RuntimeError:
                print("Raw data have been downloaded previously.")
                
            try:
                md.mmdataset(DATASET.labels, DATA_PATH)
            except RuntimeError:
                print("Labels have been downloaded previously.")
            
            # define your different modalities - refer to the filenames of the CSD files
            visual_field = 'CMU_MOSEI_VisualFacet42'
            acoustic_field = 'CMU_MOSEI_COVAREP'
            text_field = 'CMU_MOSEI_TimestampedWords'

            features = [
                text_field, 
                visual_field, 
                acoustic_field
            ]

            recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
            print(recipe)
            dataset = md.mmdataset(recipe)

            # we define a simple averaging function that does not depend on intervals
            def avg(intervals: np.array, features: np.array) -> np.array:
                try:
                    return np.average(features, axis=0)
                except:
                    return features

            # first we align to words with averaging, collapse_function receives a list of functions
            dataset.align(text_field, collapse_functions=[avg])

            label_field = 'CMU_MOSEI_Labels'

            # we add and align to lables to obtain labeled segments
            # this time we don't apply collapse functions so that the temporal sequences are preserved
            label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
            dataset.add_computational_sequences(label_recipe, destination=None)
            dataset.align(label_field)

            # obtain the train/dev/test splits - these splits are based on video IDs
            train_split = DATASET.standard_folds.standard_train_fold
            dev_split = DATASET.standard_folds.standard_valid_fold
            test_split = DATASET.standard_folds.standard_test_fold

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            pattern = re.compile('(.*)\[.*\]')
            num_drop = 0 # a counter to count how many data points went into some processing issues

            for segment in dataset[label_field].keys():
                
                # get the video ID and the features out of the aligned dataset
                try:
                    vid = re.search(pattern, segment).group(1)
                    label = dataset[label_field][segment]['features']
                    _words = dataset[text_field][segment]['features']
                    _visual = dataset[visual_field][segment]['features']
                    _acoustic = dataset[acoustic_field][segment]['features']
                except:
                    continue

                # if the sequences are not same length after alignment, there must be some problem with some modalities
                # we should drop it or inspect the data again
                if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
                    print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
                    num_drop += 1
                    continue

                # remove nan values
                label = np.nan_to_num(label)
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                actual_words = []
                words = []
                visual = []
                acoustic = []
                for i, word in enumerate(_words):
                    if word[0] != b'sp':
                        actual_words.append(word[0].decode('utf-8'))
                        words.append(word2id[word[0].decode('utf-8')]) # SDK stores strings as bytes, decode into strings here
                        visual.append(_visual[i, :])
                        acoustic.append(_acoustic[i, :])

                words = np.asarray(words)
                visual = np.asarray(visual)
                acoustic = np.asarray(acoustic)

                # z-normalization per instance and remove nan/infs
                visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
                acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

                if vid in train_split:
                    train.append(((words, visual, acoustic, actual_words), label, segment))
                elif vid in dev_split:
                    dev.append(((words, visual, acoustic, actual_words), label, segment))
                elif vid in test_split:
                    test.append(((words, visual, acoustic, actual_words), label, segment))
                else:
                    print(f"Found video that doesn't belong to any splits: {vid}")
                

            print(f"Total number of {num_drop} datapoints have been dropped.")

            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            torch.save((pretrained_emb, word2id), CACHE_PATH)

            # Save pickles
            to_pickle(train, DATA_PATH + '/bbfn_train.pkl')
            to_pickle(dev, DATA_PATH + '/bbfn_dev.pkl')
            to_pickle(test, DATA_PATH + '/bbfn_test.pkl')

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "valid":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()

class UR_FUNNY:
    def __init__(self, config):

        
        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = torch.load(CACHE_PATH)

        except:


            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)


            data_folds=load_pickle(DATA_PATH + '/data_folds.pkl')
            train_split=data_folds['train']
            dev_split=data_folds['dev']
            test_split=data_folds['test']

            

            word_aligned_openface_sdk=load_pickle(DATA_PATH + "/openface_features_sdk.pkl")
            word_aligned_covarep_sdk=load_pickle(DATA_PATH + "/covarep_features_sdk.pkl")
            word_embedding_idx_sdk=load_pickle(DATA_PATH + "/word_embedding_indexes_sdk.pkl")
            word_list_sdk=load_pickle(DATA_PATH + "/word_list.pkl")
            humor_label_sdk = load_pickle(DATA_PATH + "/humor_label_sdk.pkl")

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            num_drop = 0 # a counter to count how many data points went into some processing issues

            # Iterate over all possible utterances
            for key in humor_label_sdk.keys():

                label = np.array(humor_label_sdk[key], dtype=int)
                _word_id = np.array(word_embedding_idx_sdk[key]['punchline_embedding_indexes'])
                _acoustic = np.array(word_aligned_covarep_sdk[key]['punchline_features'])
                _visual = np.array(word_aligned_openface_sdk[key]['punchline_features'])


                if not _word_id.shape[0] == _acoustic.shape[0] == _visual.shape[0]:
                    num_drop += 1
                    continue

                # remove nan values
                label = np.array([np.nan_to_num(label)])[:, np.newaxis]
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)


                actual_words = []
                words = []
                visual = []
                acoustic = []
                for i, word_id in enumerate(_word_id):
                    word = word_list_sdk[word_id]
                    actual_words.append(word)
                    words.append(word2id[word])
                    visual.append(_visual[i, :])
                    acoustic.append(_acoustic[i, :])

                words = np.asarray(words)
                visual = np.asarray(visual)
                acoustic = np.asarray(acoustic)

                # z-normalization per instance and remove nan/infs
                visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
                acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

                if key in train_split:
                    train.append(((words, visual, acoustic, actual_words), label))
                elif key in dev_split:
                    dev.append(((words, visual, acoustic, actual_words), label))
                elif key in test_split:
                    test.append(((words, visual, acoustic, actual_words), label))
                else:
                    print(f"Found video that doesn't belong to any splits: {key}")

            print(f"Total number of {num_drop} datapoints have been dropped.")
            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            torch.save((pretrained_emb, word2id), CACHE_PATH)

            # Save pickles
            to_pickle(train, DATA_PATH + '/train.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "valid":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()