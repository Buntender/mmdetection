# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet.core import encode_mask_results
from robustdetector.utils.daedalus_loss import DaedalusLoss, outputdecode
import os
import numpy as np

#TODO change to real daedalus
#TODO trim pycharm config

#CLEAN
# INIT_K = 2000
# LR = 1e-2
# gamma = 0.05

# INIT_K = 1000
# LR = 1e-2
# gamma = 0.1

INIT_K = 200
LR = 1e-2
gamma = 0.2

#ROBUST
# INIT_K = 10
# LR = 1e-1
# gamma = 0.05

# INIT_K = 5
# LR = 1e-1
# gamma = 0.1

# INIT_K = 0.5
# LR = 1e-1
# gamma = 0.2

UPPERBOUND = 1e5
LOWERBOUND = 0

BINARYSEARCHSTEP = 7
MAXATTACKITER = 2000

earlystop = 0.995

def daedalus_single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    **kwargs):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    loss = DaedalusLoss()

    perturbation_dir = os.path.join(out_dir, 'perturbation')
    if not os.path.exists(perturbation_dir):
        os.makedirs(perturbation_dir)

    std = None
    mean = None
    for i, data in enumerate(data_loader):
        if std == None:
            std = torch.tensor(data['img_metas'][0].data[0][0]['img_norm_cfg']['std']).view(1, -1, 1, 1).cuda()
            mean = torch.tensor(data['img_metas'][0].data[0][0]['img_norm_cfg']['mean']).view(1, -1, 1, 1).cuda()

        perturb = torch.zeros(data['img'][0].data[0].shape).cuda()
        arctanori = torch.arctanh((data['img'][0].data[0].clone().detach().cuda() * std + mean) * 2 / 255 - 1)

        upperbound = (torch.ones(data['img'][0].data[0].size(0)) * UPPERBOUND).cuda()
        lowerbound = (torch.ones(data['img'][0].data[0].size(0)) * LOWERBOUND).cuda()
        k = (torch.ones(data['img'][0].data[0].size(0)) * INIT_K).cuda()
        init_pred = None

        for attach_epoch in range(BINARYSEARCHSTEP):
            advoptimizer = torch.optim.Adam([perturb], lr=LR)
            perturb.requires_grad_()
            last_losstotal = 1e5

            if attach_epoch == BINARYSEARCHSTEP - 1 and upperbound != UPPERBOUND:
                k = upperbound.clone()

            for r in range(MAXATTACKITER):
                data['img'][0].data[0] = (((torch.tanh(arctanori + perturb) + 1) * 255 / 2 - mean) / std).cpu()

                datawrapper = lambda x: {'img': x, 'img_metas': data['img_metas'][0].data[0], 'return_raw': True}
                pred = loss.forward(outputdecode(model, model(**datawrapper(data['img'][0].data[0]))), None)
                l2loss = torch.sum(torch.pow(torch.tanh(arctanori + perturb) - torch.tanh(arctanori), 2)) / (255 * 255 * 2 * 2)

                losstotal = pred + k * l2loss

                advoptimizer.zero_grad()
                losstotal.backward()
                advoptimizer.step()

                if r % 200 == 0:
                    if losstotal > last_losstotal * earlystop:
                        break

                    last_losstotal = losstotal.clone().detach()
                    if init_pred == None:
                        init_pred = pred.clone().detach()

            if pred < init_pred * (1 - gamma):
                lowerbound = k.clone()
                if upperbound == UPPERBOUND:
                    k = k*2
                else:
                    k = (upperbound + lowerbound) / 2
            else:
                upperbound = k.clone()
                k = (upperbound + lowerbound) / 2

        with torch.no_grad():
            data['img'][0].data[0].detach_()
            datarefine = {'img': [data['img'][0].data[0]], 'img_metas': [data['img_metas'][0].data[0]]}
            # result = model(return_loss=False, rescale=True, **datarefine)
            result = model(return_loss=False, rescale=True, **datarefine)

        save_patch_name = os.path.join(perturbation_dir, data['img_metas'][0].data[0][0]['ori_filename'].split('/')[-1].split('.')[0]) + '.npy'
        print("save img as ", save_patch_name)
        np.save(save_patch_name, (torch.tanh(arctanori + perturb) - torch.tanh(arctanori)).detach().cpu().numpy())

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results