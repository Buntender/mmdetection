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

@RUNNERS.register_module()
class FreeRobustRunner(EpochBasedRunner):
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

            perturb = (self.data_batch['img'].data[0].new(self.data_batch['img'].data[0].size()).uniform_(-2, 2).to(self.model.device) / torch.tensor(self.data_batch['img_metas'].data[0][0]['img_norm_cfg']['std']).view(1, -1, 1, 1).to(self.model.device))
            ori = self.data_batch['img'].data[0].clone().detach().to(self.model.device)
            # perturb = self.data_batch['img'].data[0].new(self.data_batch['img'].data[0].size()).uniform_(-2, 2).to(self.model.src_device_obj)
            # ori = self.data_batch['img'].data[0].clone().detach().to(self.model.src_device_obj)

            # for i in range(10):
            for i in range(max(min((self.epoch - 10), 10), 1)):
                self.call_hook('before_train_iter')
                self.data_batch['img'].data[0] = (ori + perturb).cpu()
                self.data_batch['img'].data[0] = self.data_batch['img'].data[0].detach()
                self.data_batch['img'].data[0].requires_grad_()
                self.run_iter(data_batch, train_mode=True, **kwargs)
                self.optimizer.zero_grad()
                # self.outputs['loss'].backward(retain_graph=True)
                self.outputs['loss'].backward()

                perturb = perturbupdater(perturb, self.data_batch['img'].data[0].grad.to(self.model.device), ori, self.data_batch['img_metas'].data[0][0]['img_norm_cfg'])

                self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
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

            perturb = (self.data_batch['img'].data[0].new(self.data_batch['img'].data[0].size()).uniform_(-2, 2).to(self.model.device) / torch.tensor(self.data_batch['img_metas'].data[0][0]['img_norm_cfg']['std']).view(1, -1, 1, 1).to(self.model.device))
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
class FreeRobustOptimizerHook(OptimizerHook):
    def after_train_iter(self, runner):
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()