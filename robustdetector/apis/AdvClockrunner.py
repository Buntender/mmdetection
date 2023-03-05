import torch
from torch.optim import lr_scheduler
from mmcv.runner import EpochBasedRunner, RUNNERS
import time
from mmcv.runner import OptimizerHook
from mmcv.runner.hooks import HOOKS, Hook
from typing import (Any, Dict, Union)
import mmcv
from robustdetector.utils.patch import load_patch, save_patch
from robustdetector.utils.adv_clock.patch_gen import PatchApplier, PatchTransformer, SmoothTV
import torch.nn.functional as F

from itertools import chain
import os

# lr = 1
lr = 1e-2
momentum = 0.9
target_class = 14
img_w = 72
img_h = 120

#TODO make tidy
#TODO test on single/multi card
@RUNNERS.register_module()
class AdvClockRunner(EpochBasedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model.training:
            self.patch = torch.nn.Parameter(torch.rand(1, 3, img_h, img_w), requires_grad=True) / 1000
        # testing
        else:
            self.patch = load_patch()

        self.patch = self.patch.cuda().detach()
        self.patch.requires_grad_()
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()

        self.patchoptimizer = torch.optim.SGD([self.patch], lr=lr, momentum=momentum)
        self.patchscheduler = lr_scheduler.StepLR(self.patchoptimizer, step_size=25, gamma=0.1)

        self.TVLoss = SmoothTV()
        self.loss = AdvClockLoss()

        self.dir = '/'.join(kwargs['meta']['config'].split('resume_from')[1].split('\n')[0].split('\'')[1].split('/')[:-1]) + '_Patch'
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir)

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        self.outputs = {"loss":None, "log_vars":{}}

        curtime = time.time()

        std = None
        mean = None
        for i, data_batch in enumerate(self.data_loader):
            if std == None:
                std = torch.tensor(data_batch['img_metas'].data[0][0]['img_norm_cfg']['std']).view(1, -1, 1, 1).cuda()
                mean = torch.tensor(data_batch['img_metas'].data[0][0]['img_norm_cfg']['mean']).view(1, -1, 1, 1).cuda()

            self._inner_iter = i

            bsz, _, height, width = data_batch['img'].data[0].shape

            lab_batch = []
            for item in range(len(data_batch['gt_labels'].data[0])):
                bboxinimg = []
                for obj in range(len(data_batch['gt_labels'].data[0][item])-1, -1, -1):
                    if data_batch['gt_labels'].data[0][item][obj] == target_class:
                        bboxinimg.append(data_batch['gt_bboxes'].data[0][item][obj])
                if len(bboxinimg) != 0:
                    lab_batch.append(torch.stack(bboxinimg))
                else:
                    lab_batch.append(torch.zeros([0,4]))

            if torch.cat(lab_batch).size(0) == 0:
                continue

            adv_batch = self.patch_transformer(
                self.patch, torch.cat(lab_batch).cuda(), height, width,
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

            img = data_batch['img'].data[0].cuda() * std + mean
            data_batch['img'].data[0] = self.patch_applier(img, adv_batch.cuda(), bbox2img)
            data_batch['img'].data[0] = ((data_batch['img'].data[0] - mean) / std).cpu()

            # plt.imshow(adv_batch[0].cpu().detach().numpy().transpose(1,2,0))
            # plt.imshow(data_batch['img'].data[0][0].detach().numpy().transpose(1,2,0))
            # plt.show()

            # for item in range(len(data_batch['gt_labels'].data[0])):
            #     # if not torch.any(data_batch['gt_labels'].data[0][item] == 14):
            #     for obj in range(len(data_batch['gt_labels'].data[0][item])-1, -1, -1):
            #         if data_batch['gt_labels'].data[0][item][obj] == target_class:
            #             data_batch['gt_labels'].data[0][item] = torch.cat([data_batch['gt_labels'].data[0][item][:obj], data_batch['gt_labels'].data[0][item][obj+1:]]).detach()
            #             data_batch['gt_bboxes'].data[0][item] = torch.cat(
            #                 [data_batch['gt_bboxes'].data[0][item][:obj], data_batch['gt_bboxes'].data[0][item][obj + 1:]]).detach()

            # for item in range(len(data_batch['gt_labels'].data[0])):
            #     # if not torch.any(data_batch['gt_labels'].data[0][item] == 14):
            #     for obj in range(len(data_batch['gt_labels'].data[0][item])-1, -1, -1):
            #         if data_batch['gt_labels'].data[0][item][obj] != target_class:
            #             data_batch['gt_labels'].data[0][item] = torch.cat([data_batch['gt_labels'].data[0][item][:obj], data_batch['gt_labels'].data[0][item][obj+1:]]).detach()
            #             data_batch['gt_bboxes'].data[0][item] = torch.cat(
            #                 [data_batch['gt_bboxes'].data[0][item][:obj], data_batch['gt_bboxes'].data[0][item][obj + 1:]]).detach()


            # for k in range(len(self.data_batch['gt_labels'].data[0])):
            #     data_batch['gt_labels'].data[0][k] = torch.ones(data_batch['gt_labels'].data[0][k].size(), dtype=torch.int64) * target_class

            # self.data_batch = copy.deepcopy(data_batch)
            # self.data_batch['img'].data[0].detach_()
            # [item.detach_() for item in self.data_batch['gt_labels'].data[0]]
            # [item.detach_() for item in self.data_batch['gt_bboxes'].data[0]]

            # for item in range(len(data_batch['gt_labels'].data[0])):
            #     for obj in range(len(data_batch['gt_labels'].data[0][item])-1, -1, -1):
            #         data_batch['gt_labels'].data[0][item] = torch.tensor([], dtype=torch.int64)
            #         data_batch['gt_bboxes'].data[0][item] = torch.zeros([0,4])

            self.data_batch = data_batch

            self.call_hook('before_train_iter')

            # for i in range(10):
            # # for i in range(max(min((self.epoch - 12) // 2, 12), 1)):

            # self.data_batch['img'].data[0].requires_grad_()
            # self.run_iter(data_batch, train_mode=True, **kwargs)
            self.attack_iter(data_batch, train_mode=True, **kwargs)

            # print(f"after attack:{time.time() - curtime}")
            # curtime = time.time()

            # self.patchoptimizer.zero_grad()
            # (-self.outputs['loss']).backward()
            # self.outputs['loss'].backward()

            self.outputs['loss'].backward()

            # self.patch.grad.clamp_(min=-1e-2, max=1e-2)
            self.patch.grad.clamp_(min=-1e-3, max=1e-3)

            self.patchoptimizer.step()
            self.patch.detach_().clip_(0, 1)
            self.patch.requires_grad_()

            # print(f"after backprop:{time.time() - curtime}")
            # curtime = time.time()

            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        save_patch(self.patch, self.dir, self.epoch)
        self.patchscheduler.step()
        self._epoch += 1

    def attack_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
        self.data_batch['img'].data[0].requires_grad_()
        datawrapper = lambda x: {'img': x, 'img_metas': self.data_batch['img_metas'].data[0], 'return_raw': True}

        pred = self.model(**datawrapper(self.data_batch['img'].data[0]))
        loss = self.loss.forward(outputdecode(self.model, pred), None)

        dummy = sum([torch.sum(layer) for layer in pred[1]]) * 0

        tvloss = self.TVLoss(self.patch)

        # self.outputs['loss'] = loss + tvloss
        self.outputs['loss'] = loss + dummy + tvloss * max(self.epoch - 40, 0) * 0.01
        # self.outputs['loss'] = loss + dummy + tvloss * max(self.epoch - 65, 0) * 0.01
        self.outputs['log_vars']['adv loss'] = float(loss)
        self.outputs['log_vars']['tv loss'] = float(tvloss)

        self.log_buffer.update(self.outputs['log_vars'])

    def register_optimizer_hook(
            self, optimizer_config: Union[Dict, Hook, None]) -> None:
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'AdvClockOptimizerHook')
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook, priority='ABOVE_NORMAL')

#TODO delete the hook
#TODO stop saving models
@HOOKS.register_module()
class AdvClockOptimizerHook(OptimizerHook):
    def after_train_iter(self, runner):
        pass

class AdvClockLoss():
    bboxes = [5776, 2166, 600, 150, 36, 4]
    weight = [0.1, 0.3, 0.5, 0.7, 0.9, 1]

    def __init__(self):
        self.weights = [[self.weight[item]] * self.bboxes[item] for item in range(len(self.bboxes))]
        self.weights = torch.tensor(list(chain(*self.weights)))

    def forward(self, predictions, obj):
        lossperbox = torch.pow((F.softmax(predictions, dim=-1)[:, :, -1] - 0.75).clip(max=0), 2)
        self.weights = self.weights.to(lossperbox.device)
        return torch.sum(lossperbox * self.weights)

def outputdecode(model, res):
    cls_score_img_list = [layer.permute(0, 2, 3, 1).reshape(layer.size(0), -1, model.module.bbox_head.cls_out_channels) for layer in res[0]]
    return torch.cat(cls_score_img_list, dim=1)