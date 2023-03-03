import numpy as np
from sklearn.manifold import TSNE
import torch
import os
from os import path
import itertools

def scaner_file(url):
    file = os.listdir(url)
    reslist = []
    for f in file:
        real_url = path.join(url, f)
        if path.isfile(real_url):
            reslist.append(path.abspath(real_url))
        elif path.isdir(real_url):
            reslist.append(scaner_file(real_url))
    return reslist

filefolders = scaner_file('/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature')
files = [torch.stack([torch.load(filedir,map_location=torch.device('cpu'))[-1].reshape(-1) for filedir in folder])
        for folder in filefolders]

# tags = [np.ones(files[0].size(0)) * i for i in range(len(files))]
tags = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
tags = list(itertools.chain.from_iterable([[tags[i]] * len(files[i]) for i in range(len(files))]))

features = torch.cat(files).numpy()
# tags = np.concatenate(tags)

model = TSNE()
np.set_printoptions(suppress=True)
featuresthin = model.fit_transform(features) # 将X降维(默认二维)后保存到Y中

from matplotlib import pyplot as plt
plt.scatter(featuresthin[:,0], featuresthin[:,1], 5, c=tags, label=tags) # labels为每一行对应标签，20为标记大小
plt.legend(loc='upper right')
plt.savefig("transH.png") #保存图片
plt.show()
