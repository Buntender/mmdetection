import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

totalpert = 0
perturbation_dir = "work_dirs/ssd300_voc_daedalus_gamma020_res/perturbation"
for dirpath, dirnames, filenames in os.walk(perturbation_dir):
    for filename in filenames:
        perturb = torch.tensor(np.load(os.path.join(perturbation_dir, filename))) / 2
        totalpert += torch.sum(torch.pow(perturb, 2))

        # from robustdetector.utils.adv_clock.patch_gen import MedianPool2d
        # medianpooler = MedianPool2d(5, same=True)
        # plt.imshow(medianpooler(patch).cpu().squeeze().numpy().transpose(1, 2, 0))
        # plt.show()

        # patch_img = Image.fromarray((perturb*20+0.5).squeeze().numpy().transpose(1, 2, 0))
        # patch_img.show()

totalpert /= len(filenames)
print(totalpert)