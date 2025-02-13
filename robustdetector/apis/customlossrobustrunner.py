import torch
from mmcv.runner import EpochBasedRunner, RUNNERS
import time
from typing import (Any)
from robustdetector.utils.daedalus_loss import outputdecode
from robustdetector.utils.robustutils import perturbupdater

#TODO change to FGSM runner
@RUNNERS.register_module()
class CustomLossRobustRunner(EpochBasedRunner):
    def __init__(self, *kargs, **kwargs):
        self.loss = kwargs.pop('custom_loss')
        super(CustomLossRobustRunner, self).__init__(*kargs, **kwargs)
        # import_module('robustdetector.apis.'+self.loss)
        if self.loss != None:
            self.loss = globals()[self.loss]


    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        mean = None
        std = None
        for i, data_batch in enumerate(self.data_loader):
            if std == None:
                mean = torch.tensor(data_batch['img_metas'].data[0][0]['img_norm_cfg']['mean']).view(1, -1, 1, 1).to(self.model.device)
                std = torch.tensor(data_batch['img_metas'].data[0][0]['img_norm_cfg']['std']).view(1, -1, 1, 1).to(self.model.device)

            self.data_batch = data_batch
            self._inner_iter = i

            #TODO universal noise initializer
            perturb = self.data_batch['img'].data[0].new(self.data_batch['img'].data[0].size()).uniform_(-2, 2).to(self.model.device)
            ori = self.data_batch['img'].data[0].clone().detach().to(self.model.device)

            # for i in range(10):
            for i in range(max(min(self.epoch - 5, 10), 1)):
                self.call_hook('before_train_iter')

                perturb = perturbupdater(perturb, self.attack_iter(data_batch, train_mode=True, **kwargs), ori, mean, std)

                self.data_batch['img'].data[0] = ori + perturb
                self.data_batch['img'].data[0].detach_()

                self.run_iter(data_batch, train_mode=True, **kwargs)

                self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def attack_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
        self.data_batch['img'].data[0].requires_grad_()
        datawrapper = lambda x: {'img': x, 'img_metas': self.data_batch['img_metas'].data[0], 'return_raw': True}
        pred = self.loss.forward(outputdecode(self.model, self.model(**datawrapper(self.data_batch['img'].data[0]))[0]), None)
        pred.backward()

        return self.data_batch['img'].data[0].grad