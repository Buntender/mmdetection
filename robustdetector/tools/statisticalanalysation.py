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

filefolders = scaner_file('/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/loc')
files = [torch.stack([torch.load(filedir,map_location=torch.device('cuda'))[-1].reshape(-1) for filedir in folder])
        for folder in filefolders]
ori = files[0]
files = files[1:]

# ori = scaner_file('/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/ori')
# ori = torch.stack([torch.load(filedir, map_location=torch.device('cpu'))[-1].reshape(-1) for filedir in ori])

distances = [torch.pow(feature - ori, 2).mean() for feature in files]
print(distances)