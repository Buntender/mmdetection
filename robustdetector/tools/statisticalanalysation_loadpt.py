import torch

filedir = '/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/StatisticalAnalysis/epochwise_differenttestdistance/vocclean_cls_vocclean1_locdist.pt'
print(torch.load(filedir, map_location=torch.device('cpu')))
