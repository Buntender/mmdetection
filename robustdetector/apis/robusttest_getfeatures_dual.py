# Copyright (c) OpenMMLab. All rights reserved.
import functools
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from robustdetector.apis.robustutils import perturbupdater

def _robust_single_gpu_test_dual(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    backpropfunc = None,
                    dir_name = None,
                    model2 = None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        perturb = data['img'][0].data[0].new(data['img'][0].data[0].size()).uniform_(-2, 2).cuda()
        ori = data['img'][0].data[0].clone().detach().cuda()

        for r in range(10):
            data['img'][0].data[0] = (ori + perturb).cpu()
            data['img'][0].data[0] = data['img'][0].data[0].detach()
            data['img'][0].data[0].requires_grad_()

            loss = model2(img=data['img'][0], img_metas=data['img_metas'][0], gt_bboxes=data['gt_bboxes'][0], gt_labels=data['gt_labels'][0])
            # loss = model(**data)
            backpropfunc(loss, model2)
            perturb = perturbupdater(perturb, data['img'][0].data[0].grad.cuda(), ori, data['img_metas'][0].data[0][0]['img_norm_cfg'])


        with torch.no_grad():
            datarefine = {'img': data['img'][0], 'img_metas': data['img_metas'][0]}
            result = model(return_raw = True, rescale=True, **datarefine)[1]
            filename = data['img_metas'][0].data[0][0]['ori_filename']
            filename = filename.split('/')[1].split('.')[0]
            torch.save(result[3:], dir_name + filename + '.pth')

robust_single_gpu_getfeature_dual = functools.partial(_robust_single_gpu_test_dual, backpropfunc= lambda loss, model : model.module._parse_losses(loss)[0].backward())
clsloss_single_gpu_getfeature_dual = functools.partial(_robust_single_gpu_test_dual, backpropfunc= lambda loss, model : loss['loss_cls'][0].backward())
bboxloss_single_gpu_getfeature_dual = functools.partial(_robust_single_gpu_test_dual, backpropfunc= lambda loss, model : loss['loss_bbox'][0].backward())

# robust_multi_gpu_getfeature = functools.partial(_robust_multi_gpu_test, backpropfunc= lambda loss, model : model.module._parse_losses(loss)[0].backward())
# clsloss_multi_gpu_getfeature = functools.partial(_robust_multi_gpu_test, backpropfunc= lambda loss, model : (loss['loss_cls'][0] + loss['loss_bbox'][0] * 0).backward())
# bboxloss_multi_gpu_getfeature = functools.partial(_robust_multi_gpu_test, backpropfunc= lambda loss, model : (loss['loss_cls'][0] * 0 + loss['loss_bbox'][0]).backward())