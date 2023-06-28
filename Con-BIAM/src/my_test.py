import numpy as np
from tqdm import tqdm_notebook

word_emb_path = 'D:\python\PyCharm 2022.2.1\projects\BBFN\home\yingting\Glove\glove.840B.300d.txt'
embedding_vocab = 2196017
f = open(word_emb_path, 'r', encoding='UTF-8')
for line in tqdm_notebook(f, total=embedding_vocab):
    # 读取每一行句子， 并且去掉空格
    print("line = ")
    print(line)
    print(type(line))
    content = line.strip().split()
    print(type(content))
    vector = np.asarray(list(map(lambda x: float(x), content[-300:])))
    print("vector = " )
    print(vector[0])
    print(type(vector))
    print(vector.shape)
    word = ' '.join(content[:-300])
    print("word = " + word)
    # if word in w2i:
    #     idx = w2i[word]
    #     emb_mat[idx, :] = vector
    #     found += 1
    exit()