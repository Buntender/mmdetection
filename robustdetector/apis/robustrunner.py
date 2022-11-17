import torch
from mmcv.runner import EpochBasedRunner, RUNNERS
from typing import Any
import time
from mmcv.runner import OptimizerHook
from mmcv.runner.hooks import HOOKS, Hook
from typing import (Dict, List, Optional, Union)
import mmcv

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
            perturb = self.data_batch['img'].data[0].new(self.data_batch['img'].data[0].size()).uniform_(-1, 1)
            ori = self.data_batch['img'].data[0]
            # for i in range(10):
            for i in range(max(min(self.epoch - 5, 10), 1)):
                self.call_hook('before_train_iter')
                self.data_batch['img'].data[0] = ori + perturb
                self.data_batch['img'].data[0].detach_()
                self.data_batch['img'].data[0].requires_grad_()
                self.run_iter(data_batch, train_mode=True, **kwargs)
                self.optimizer.zero_grad()
                # self.outputs['loss'].backward(retain_graph=True)
                self.outputs['loss'].backward()

                grad = self.data_batch['img'].data[0].grad.clone()
                perturb += grad.sign()
                perturb = perturb.clamp(-8, 8)
                perturb.detach_()

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

            perturb = self.data_batch['img'].data[0].new(self.data_batch['img'].data[0].size()).uniform_(-1, 1)
            ori = self.data_batch['img'].data[0]
            for i in range(10):
                self.data_batch['img'].data[0] = ori + perturb
                self.data_batch['img'].data[0].detach_()
                self.data_batch['img'].data[0].requires_grad_()
                self.run_iter(data_batch, train_mode=True, **kwargs)
                self.optimizer.zero_grad()
                # self.outputs['loss'].backward(retain_graph=True)
                self.outputs['loss'].backward()

                grad = self.data_batch['img'].data[0].grad.clone()
                perturb += grad.sign()
                perturb = perturb.clamp(-8, 8)
                perturb.detach_()

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