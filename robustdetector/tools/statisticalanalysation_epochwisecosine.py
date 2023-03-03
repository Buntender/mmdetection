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

dir_name = '/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/StatisticalAnalysis/epochwise_cleanrobustdistance/cls_loc_clean/'
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)

for epoch in range(1, 25):
    # filefolderslhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/loc_gentest_bimodel/epoch_{epoch}')
    # filefoldersrhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/loc_old')

    # filefolderslhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/loc_gentest_bimodel/epoch_{epoch}')
    # filefolderslhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/cls_gentest_bimodel_FreeRobust/epoch_{epoch}')
    # filefoldersrhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/loc_gentest_bimodel_FreeRobust/epoch_{epoch}')

    filefolderslhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/cls_gentest_bimodel/epoch_{epoch}')
    filefoldersrhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/loc_gentest_bimodel/epoch_{epoch}')

    # filefoldersrhs = scaner_file(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/cls_old')

    # filefoldersrhs = filefoldersrhs[:1]

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
    torch.save(distances, dir_name + f'distances_{epoch}.pt')