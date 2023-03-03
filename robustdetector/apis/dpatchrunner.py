import torch
from mmcv.runner import EpochBasedRunner, RUNNERS
from typing import Any
import time
from mmcv.runner import OptimizerHook
from mmcv.runner.hooks import HOOKS, Hook
import logging
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    no_type_check)
import mmcv
from robustdetector.apis.robustutils import perturbupdater
from robustdetector.apis.patch import add_patch, load_patch, save_patch

lr = 1e-2
momentum = 0.9
weight_decay = 0.0005
target_class = 14
img_w = 300
img_h = 300

@RUNNERS.register_module()
class DPatchRunner(EpochBasedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model.training:
            self.patch = torch.nn.Parameter(torch.rand(1, 3, img_w, img_h), requires_grad=True)
        # testing
        else:
            self.patch = load_patch()

        self.patch = self.patch.cuda().detach()
        self.patch.requires_grad_()
        self.patchoptimizer = torch.optim.SGD([self.patch], lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i

            for k in range(self.data_batch['img'].data[0].size(0)):
                self.data_batch['img'].data[0] = add_patch(self.data_batch['img'].data[0].cuda(), self.patch).cpu()

            # for k in range(len(self.data_batch['gt_labels'].data[0])):
            #     data_batch['gt_labels'].data[0][k] = torch.ones(data_batch['gt_labels'].data[0][k].size(), dtype=torch.int64) * target_class

            # for i in range(10):
            # for i in range(max(min((self.epoch - 12) // 2, 12), 1)):
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.patchoptimizer.zero_grad()
            # self.outputs['loss'].backward(retain_graph=True)
            # self.outputs['loss'].backward()

            (-self.outputs['loss']).backward()

            self.patchoptimizer.step()
            self.patch.clip(0, 1)
            # self.patch = torch.clip(self.patch, 0, 1).detach()

            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        save_patch(self.patch, self.epoch)
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_val_iter')

            perturb = self.data_batch['img'].data[0].new(self.data_batch['img'].data[0].size()).uniform_(-2, 2).to(self.model.device)
            ori = self.data_batch['img'].data[0].clone().cuda().to(self.model.device)

            for i in range(10):
                self.data_batch['img'].data[0] = (ori + perturb).cpu()
                self.data_batch['img'].data[0].detach_()
                self.data_batch['img'].data[0].requires_grad_()
                self.run_iter(data_batch, train_mode=True, **kwargs)
                self.optimizer.zero_grad()
                # self.outputs['loss'].backward(retain_graph=True)
                self.outputs['loss'].backward()

                perturb = perturbupdater(perturb, self.data_batch['img'].data[0].grad.to(self.model.device), ori, self.data_batch['img_metas'][0].data[0][0]['img_norm_cfg'])

            with torch.no_grad():
                self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')
            del self.data_batch
        self.call_hook('after_val_epoch')

    def register_optimizer_hook(
            self, optimizer_config: Union[Dict, Hook, None]) -> None:
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'FreeRobustOptimizerHook')
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook, priority='ABOVE_NORMAL')


@HOOKS.register_module()
class DPatchOptimizerHook(OptimizerHook):
    def after_train_iter(self, runner):
        pass
        # if self.grad_clip is not None:
        #     grad_norm = self.clip_grads(runner.model.parameters())
        #     if grad_norm is not None:
        #         # Add grad norm to the logger
        #         runner.log_buffer.update({'grad_norm': float(grad_norm)},
        #                                  runner.outputs['num_samples'])
        # runner.optimizer.step()