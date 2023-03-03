import numpy as np
from sklearn.manifold import TSNE
import torch
import os
from os import path
import random

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
tags = [np.ones(files[0].size(0)) * i for i in range(len(files))]

features = torch.cat(files).numpy()
tags = np.concatenate(tags).astype(int)

model = TSNE(n_iter=3000)
np.set_printoptions(suppress=True)
featuresthin = model.fit_transform(features) # 将X降维(默认二维)后保存到Y中

sample_num = 5000
sample_list = [i for i in range(features.shape[0])] # [0, 1, 2, 3, 4, 5, 6, 7]
sample_list = random.sample(sample_list, sample_num) #随机选取出了 [3, 4, 2, 0]

featuresthin = featuresthin[sample_list,:]
tags = tags[sample_list]

from matplotlib import pyplot as plt
scatter = plt.scatter(featuresthin[:,0], featuresthin[:,1], 5, c=tags, cmap=plt.cm.jet) # labels为每一行对应标签，20为标记大小
legend = plt.legend(scatter.legend_elements()[0], ["ori", "cls", "loc", "con"],
                    loc="upper left", title="class")

plt.savefig("transH.png") #保存图片
plt.show()
