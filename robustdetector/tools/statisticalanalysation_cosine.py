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

test = 1
catagory1 = 'loc'
catagory2 = 'loc'

# filefolderslhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/{catagory1}_old')
# filefoldersrhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/voc_test{test}/{catagory2}_old')
filefolderslhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/voc_test0/{catagory1}_old')
filefoldersrhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/voc_test1/{catagory2}_old')

# filefolderslhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/{catagory1}_old')
# filefoldersrhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/{catagory2}_old')

# filefoldersrhs = filefolderslhs[2:4]
filefolderslhs = filefolderslhs[:2]
# filefoldersrhs = filefoldersrhs[:2]

# filefolderslhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/voc_Freerobust/cls_old')
# filefoldersrhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/voc_Freerobust0/cls_old')

fileslhs = [torch.stack([torch.load(filedir,map_location=torch.device('cpu'))[-1].reshape(-1) for filedir in folder]).cuda()
        for folder in filefolderslhs]
filesrhs = [torch.stack([torch.load(filedir,map_location=torch.device('cpu'))[-1].reshape(-1) for filedir in folder]).cuda()
        for folder in filefoldersrhs]

ori = scaner_file('/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/ori')
ori = torch.stack([torch.load(filedir, map_location=torch.device('cuda'))[-1].reshape(-1) for filedir in ori])

difflhs = [file - ori for file in fileslhs]
diffrhs = [file - ori for file in filesrhs]

distances = [(difflhs[i] * diffrhs[i]).sum() / torch.pow(torch.pow(difflhs[i], 2).sum() * torch.pow(diffrhs[i], 2).sum(), 0.5) for i in range(len(difflhs))]

print(distances)

# dir_name = '/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/StatisticalAnalysis/epochwise_differenttestdistance/'
# if not os.path.isdir(dir_name):
#     os.makedirs(dir_name)
# torch.save(distances, dir_name + f'vocclean_{catagory1}_vocclean_{catagory2}dist.pt')
# torch.save(distances, dir_name + f'vocclean_{catagory1}_vocclean{test}_{catagory2}dist.pt')
# torch.save(distances, dir_name + 'vocrobust_vocrobust0_clsdist.pt')