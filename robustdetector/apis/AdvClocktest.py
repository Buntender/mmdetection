# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet.core import encode_mask_results

from robustdetector.utils.patch import load_patch
from robustdetector.utils.adv_clock.patch_gen import PatchApplier, PatchTransformer

target_class = 14
patch_applier = PatchApplier().cuda()
patch_transformer = PatchTransformer().cuda()

def AdcClock_single_gpu_test(model,
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
    # patch = load_patch("work_dirs/ssd300_voc_AdvClock_0301_varlr_Patch/29.npy").cuda()
    # patch = load_patch("work_dirs/ssd300_voc_AdvClock_0301_clsloss_Patch/99.npy").cuda()
    patch = load_patch("work_dirs/ssd300_voc_AdvClock_0304_Patch/26.npy").cuda()


    # import matplotlib.pyplot as plt
    # plt.imshow(patch.cpu().squeeze().numpy().transpose(1, 2, 0))

    # from robustdetector.adv_clock.patch_gen import MedianPool2d
    # medianpooler = MedianPool2d(5, same=True)
    # plt.imshow(medianpooler(patch).cpu().squeeze().numpy().transpose(1, 2, 0))

    for i, data in enumerate(data_loader):

        bsz, _, height, width = data['img'][0].data[0].shape
        lab_batch = []
        for item in range(len(data['gt_labels'][0].data[0])):
            bboxinimg = []
            for obj in range(len(data['gt_labels'][0].data[0][item]) - 1, -1, -1):
                if data['gt_labels'][0].data[0][item][obj] == target_class:
                    bboxinimg.append(data['gt_bboxes'][0].data[0][item][obj].clone().detach())

            if len(bboxinimg) != 0:
                lab_batch.append(torch.stack(bboxinimg))
            else:
                lab_batch.append(torch.zeros([0, 4]))

        if torch.cat(lab_batch).size(0) != 0:

            adv_batch = patch_transformer(
                patch, torch.cat(lab_batch).cuda(), height, width,
                rand_loc=True, scale_factor=0.22,
                cls_label=int(target_class)).mul_(255)

            bbox2img = [0] * torch.cat(lab_batch).size(0)
            sum = 0
            ptr = -1
            for cur in range(len(bbox2img)):
                while cur >= sum:
                    ptr += 1
                    sum += lab_batch[ptr].size(0)
                bbox2img[cur] = ptr

            img_std = torch.tensor(data['img_metas'][0].data[0][0]['img_norm_cfg']['std']).view(1, -1, 1, 1).cuda()
            img_mean = torch.tensor(data['img_metas'][0].data[0][0]['img_norm_cfg']['mean']).view(1, -1, 1, 1).cuda()

            img = data['img'][0].data[0].cuda() * img_std + img_mean
            data['img'][0].data[0] = patch_applier(img, adv_batch.cuda(), bbox2img)
            data['img'][0].data[0] = ((data['img'][0].data[0] - img_mean) / img_std).cpu()

        with torch.no_grad():
            datarefine = {'img': data['img'][0].data, 'img_metas': [data['img_metas'][0].data[0]]}
            # result = model(return_loss=False, rescale=True, **datarefine)
            result = model(return_loss=False, rescale=True, **datarefine)

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