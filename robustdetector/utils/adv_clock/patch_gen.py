import fnmatch
import math
import os
import sys
import time
from operator import itemgetter

import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from kornia.geometry.transform.imgwarp import get_perspective_transform, warp_perspective

import random
import kornia
import torchvision.utils as tvutils

from .utils import MedianPool2d
import robustdetector.utils.adv_clock.thinplate as tps
from robustdetector.utils.adv_clock.thinplate.pytorch import tps_grid

class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.
    Total variation loss
    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        tvcomp1 = adv_patch[:, :, 1:] - adv_patch[:, :, :-1]
        tvcomp1 = (tvcomp1 * tvcomp1 + .01).sqrt().sum()
        tvcomp2 = adv_patch[:, 1:, :] - adv_patch[:, :-1, :]
        tvcomp2 = (tvcomp2 * tvcomp2 + .01).sqrt().sum()
        tv = tvcomp1 + tvcomp2
        return tv


class SmoothTV(nn.Module):
    """TotalVariation: calculates the total variation of a patch.
    Smooth total variation loss
    """

    def __init__(self):
        super(SmoothTV, self).__init__()

    def forward(self, adv_patch):
        tvcomp1 = adv_patch[:, :, 1:] - adv_patch[:, :, :-1]
        tvcomp1 = torch.pow(tvcomp1, 2).sum()
        tvcomp2 = adv_patch[:, 1:, :] - adv_patch[:, :-1, :]
        tvcomp2 = torch.pow(tvcomp2, 2).sum()
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches
    Generate patches with different augmentations
    """

    def __init__(self, augment=True):
        super(PatchTransformer, self).__init__()
        self.augment = augment
        self.min_contrast = 0.8 if augment else 1
        self.max_contrast = 1.2 if augment else 1
        self.min_brightness = -0.1 if augment else 0
        self.max_brightness = 0.1 if augment else 0
        self.noise_factor = 0.0
        self.minangle = -5 / 180 * math.pi if augment else 0
        self.maxangle = 5 / 180 * math.pi if augment else 0
        self.medianpooler = MedianPool2d(5, same=True)
        self.distortion_max = 0.1 if augment else 0
        self.sliding = -0.05 if augment else 0

    def get_tps_thetas(self, num_images):
        c_src = np.array([
            [0.0, 0.0],
            [1., 0],
            [1, 1],
            [0, 1],
            [0.2, 0.3],
            [0.6, 0.7],
        ])

        theta_list = []
        dst_list = []
        for i in range(num_images):
            c_dst = c_src + np.random.uniform(-1, 1, (6, 2)) / 20

            theta = tps.tps_theta_from_points(c_src, c_dst)
            theta_list.append(torch.from_numpy(theta).unsqueeze(0))
            dst_list.append(torch.from_numpy(c_dst).unsqueeze(0))

        theta = torch.cat(theta_list, dim=0).float()
        dst = torch.cat(dst_list, dim=0).float()
        return theta, dst

    def get_perspective_params(self, width, height, distortion_scale):
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def forward(self, adv_patch, lab_batch, height, width, do_rotate=True, rand_loc=False, scale_factor=0.25,
                cls_label=1):

        # adv_patch = self.medianpooler(adv_patch) * 0.9 + adv_patch * 0.1
        adv_patch = self.medianpooler(adv_patch)
        patch_size = adv_patch.size(-1)

        # Determine size of padding
        pad_width = (width - adv_patch.size(-1)) / 2
        pad_height = (height - adv_patch.size(-2)) / 2

        # Make a batch of patches
        batch_size = lab_batch.size(0)
        adv_batch = adv_patch.expand(batch_size, -1, -1, -1)
        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(0, 255) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise
        adv_batch = torch.clamp(adv_batch, 0.0001, 254.999)
        # adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # perspective transformations
        dis_scale = lab_batch.size(0)
        distortion = torch.empty(dis_scale).uniform_(0, self.distortion_max)

        adv_height = adv_batch.size(-1)
        adv_width = adv_batch.size(-2)

        # tps transformation
        if self.augment:
            theta, dst = self.get_tps_thetas(dis_scale)
            img = adv_batch.clone()

            grid = tps_grid(theta.cuda(), dst.cuda(), (img.size(0), 1, adv_width, adv_height))
            adv_batch = F.grid_sample(img, grid.cuda(), padding_mode='border')
            adv_batch = adv_batch.view(lab_batch.size(0), 3, adv_width, adv_height)

        # perpective transformations with random distortion scales
        start_end = [torch.tensor(self.get_perspective_params(adv_width, adv_height, x), \
                                  dtype=torch.float).unsqueeze(0) for x in distortion]
        start_end = torch.cat(start_end, 0)
        start_points = start_end[:, 0, :, :].squeeze()
        end_points = start_end[:, 1, :, :].squeeze()

        if dis_scale == 1:
            start_points = start_points.unsqueeze(0)
            end_points = end_points.unsqueeze(0)
        try:
            M = get_perspective_transform(start_points, end_points)
            adv_batch = warp_perspective(adv_batch, M.cuda(), dsize=(adv_width, adv_height))
        except:
            print('hihi')

        mypad = nn.ConstantPad2d((int(pad_width + 0.5), int(pad_width), int(pad_height + 0.5), int(pad_height)), 0)
        adv_batch = mypad(adv_batch)

        # Rotation and rescaling transforms
        anglesize = lab_batch.size(0)
        if do_rotate:
            angle = torch.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            angle = torch.FloatTensor(anglesize).fill_(0)
        angle = angle.cuda()

        # roughly estimate the size and compute the scale
        lab_batch = lab_batch.cuda()
        target_size = scale_factor * torch.sqrt((lab_batch[:, 2] - lab_batch[:, 0]) ** 2 + (lab_batch[:, 3] - lab_batch[:, 1]) ** 2)

        # target_x = lab_batch[:, 0].view(np.prod(batch_size)) / width  # (batch_size, num_objects)
        # target_y = lab_batch[:, 1].view(np.prod(batch_size)) / height
        # target_y = target_y - 0.0

        target_x = (lab_batch[:, 2] + lab_batch[:, 0]).view(np.prod(batch_size)) / 2 / width  # (batch_size, num_objects)
        target_y = (lab_batch[:, 3] + lab_batch[:, 1]).view(np.prod(batch_size)) / 2 / height

        # shift a bit from the center
        targetoff_x = (lab_batch[:, 2] - lab_batch[:, 0]).view(np.prod(batch_size)) / width
        targetoff_y = (lab_batch[:, 3] - lab_batch[:, 1]).view(np.prod(batch_size)) / height

        off_y = (torch.FloatTensor(targetoff_y.size()).uniform_(self.sliding, 0))
        off_y = targetoff_y * off_y.cuda()
        target_y = target_y + off_y

        scale = target_size / patch_size
        scale = scale.view(anglesize)
        scale = scale.cuda()

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation,rescale matrix
        theta = torch.FloatTensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale
        theta = theta.cuda()

        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch = F.grid_sample(adv_batch, grid)

        return adv_batch


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch, bbox2img):
        advs = torch.unbind(adv_batch, 0)

        for bbox in range(len(advs)):
            img_batch[bbox2img[bbox]] = torch.where((advs[bbox] == 0), img_batch[bbox2img[bbox]], advs[bbox])
        return img_batch
