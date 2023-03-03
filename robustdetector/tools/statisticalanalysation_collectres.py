import torch
from matplotlib import pyplot as plt

line1 = []
line2 = []
line3 = []
for epoch in range(1, 25):
    line1.append(float(torch.stack(torch.load(
        f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/StatisticalAnalysis/epochwise_cls_diffepochdistance/distances_{epoch}.pt'
        , map_location=torch.device('cpu'))).mean()))

    line2.append(float(torch.stack(torch.load(
        f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/StatisticalAnalysis/epochwise_loc_diffepochdistance/distances_{epoch}.pt'
        , map_location=torch.device('cpu'))).mean()))


    line3.append(float(torch.stack(torch.load(
        f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/StatisticalAnalysis/epochwise_clsloc_diffepochdistance/distances_{epoch}.pt'
        , map_location=torch.device('cpu'))).mean()))

    # line2.append(float(torch.stack(torch.load(
    #     f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/StatisticalAnalysis/epochwise_cleanrobustdistance/cls_loc/distances_{epoch}.pt'
    #     , map_location=torch.device('cpu'))).mean()))


    # line1.append(float(torch.stack(torch.load(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/StatisticalAnalysis/epochwise_cleanrobustdistance/cls/distances_{epoch}.pt'
    #            , map_location=torch.device('cpu'))).mean()))
    #
    # line2.append(float(torch.stack(torch.load(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/StatisticalAnalysis/epochwise_cleanrobustdistance/loc/distances_{epoch}.pt'
    #            , map_location=torch.device('cpu'))).mean()))

    # line3.append(float(torch.stack(torch.load(f'/media/data4/lkz/mmdetection_stable_on_28_1124/ext_feature/StatisticalAnalysis/epochwise_cleanrobustdistance/cls_loc/distances_{epoch}.pt'
    #            , map_location=torch.device('cpu'))).mean()))

# line3 = torch.ones(24) * 0.57

plt.plot(list(range(1, 25)), line1, label='cls-cls(clean last epoch)')
plt.plot(list(range(1, 25)), line2, label='loc-loc(clean last epoch)')
plt.plot(list(range(1, 25)), line3, label='cls-loc')
# plt.plot(list(range(1, 25)), line3, linestyle='--', label='cls(clean last epoch)-loc(clean last epoch)')
plt.ylim(0,1)
plt.legend()
plt.show()