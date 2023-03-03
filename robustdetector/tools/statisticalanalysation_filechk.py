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

filefolders = scaner_file('/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/voc_test0')
for epoch in range(2):
    for round in range(2):
        print(filefolders[epoch][round][0])
        print(len(filefolders[epoch][round]))

# for epoch in range(10):
#     print(filefolders[epoch][0])
#     print(len(filefolders[epoch]))
pass