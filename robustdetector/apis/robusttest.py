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
from mmdet.apis.test import collect_results_cpu, collect_results_gpu

def _robust_single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    backpropfunc = None,
                    **kwargs):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))

    std = None
    for i, data in enumerate(data_loader):
        if std == None:
            std = torch.tensor(data['img_metas'][0].data[0][0]['img_norm_cfg']['std']).view(1, -1, 1, 1).cuda()

        perturb = data['img'][0].data[0].new(data['img'][0].data[0].size()).uniform_(-2, 2).cuda() / std
        ori = data['img'][0].data[0].clone().detach().cuda()

        for r in range(10):
            data['img'][0].data[0] = (ori + perturb).cpu()
            data['img'][0].data[0].detach_()
            data['img'][0].data[0].requires_grad_()

            loss = model(img=data['img'][0], img_metas=data['img_metas'][0], gt_bboxes=data['gt_bboxes'][0], gt_labels=data['gt_labels'][0])
            # loss = model(**data)
            backpropfunc(loss, model)
            perturb = perturbupdater(perturb, data['img'][0].data[0].grad.cuda(), ori, data['img_metas'][0].data[0][0]['img_norm_cfg'])

        with torch.no_grad():
            datarefine = {'img': [data['img'][0].data[0]], 'img_metas': [data['img_metas'][0].data[0]]}
            # result = model(return_loss=False, rescale=True, **datarefine)
            result = model(return_loss=False, rescale=True, **datarefine)
            data['img'][0].data[0].detach_()

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


def _robust_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False,
                    backpropfunc = None, **kwargs):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    std = None
    #TODO test all attack tests
    for i, data in enumerate(data_loader):
        data = model.scatter(data, None, model.device_ids)[0][0]

        if std == None:
            std = torch.tensor(data['img_metas'][0].data[0][0]['img_norm_cfg']['std']).view(1, -1, 1, 1).to(
                data['img'][0].device)

        perturb = data['img'][0].new(data['img'][0].size()).uniform_(-2, 2) / data['img_metas'][0][0]['img_norm_cfg']['std'].cuda()
        ori = data['img'][0].clone().detach()

        for r in range(10):
            data['img'][0] = (ori + perturb).cpu()
            data['img'][0].detach_()
            data['img'][0].requires_grad_()

            loss = model(img=data['img'][0], img_metas=data['img_metas'][0], gt_bboxes=data['gt_bboxes'][0], gt_labels=data['gt_labels'][0])
            # loss = model(**data)
            backpropfunc(loss, model)
            perturb = perturbupdater(perturb, data['img'][0].grad.to(perturb.device), ori, data['img_metas'][0][0]['img_norm_cfg'])

        with torch.no_grad():
            datarefine = {'img': [data['img'][0]], 'img_metas': [data['img_metas'][0]]}
            result = model(return_loss=False, rescale=True, **datarefine)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (
                        bbox_results, encode_mask_results(mask_results))

        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results

robust_single_gpu_test = functools.partial(_robust_single_gpu_test, backpropfunc= lambda loss, model : model.module._parse_losses(loss)[0].backward())
clsloss_single_gpu_test = functools.partial(_robust_single_gpu_test, backpropfunc= lambda loss, model : loss['loss_cls'][0].backward())
bboxloss_single_gpu_test = functools.partial(_robust_single_gpu_test, backpropfunc= lambda loss, model : loss['loss_bbox'][0].backward())

robust_multi_gpu_test = functools.partial(_robust_multi_gpu_test, backpropfunc= lambda loss, model : model.module._parse_losses(loss)[0].backward())
clsloss_multi_gpu_test = functools.partial(_robust_multi_gpu_test, backpropfunc= lambda loss, model : (loss['loss_cls'][0] + loss['loss_bbox'][0] * 0).backward())
bboxloss_multi_gpu_test = functools.partial(_robust_multi_gpu_test, backpropfunc= lambda loss, model : (loss['loss_cls'][0] * 0 + loss['loss_bbox'][0]).backward())