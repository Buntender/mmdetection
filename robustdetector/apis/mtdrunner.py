import torch
from mmcv.runner import EpochBasedRunner, RUNNERS
import time
from typing import (Any)
from robustdetector.utils.daedalus_loss import outputdecode
from robustdetector.utils.robustutils import FGSMupdater
import torch.distributed

#TODO change to FGSM runner

losstype = {'cls' : lambda loss : sum(loss['loss_cls']) + sum(loss['loss_bbox']) * 0,
            'bbox' : lambda loss : sum(loss['loss_cls']) * 0 + sum(loss['loss_bbox']),
            'con' : lambda loss : sum(loss['loss_cls']) + sum(loss['loss_bbox'])}

@RUNNERS.register_module()
class MTDRunner(EpochBasedRunner):
    def __init__(self, *kargs, **kwargs):
        super(MTDRunner, self).__init__(*kargs, **kwargs)

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
            self.data_batch = data_batch
            self._inner_iter = i

            if std == None:
                mean = torch.tensor(self.data_batch['img_metas'].data[0][0]['img_norm_cfg']['mean']).view(1, -1, 1, 1).to(self.model.device)
                std = torch.tensor(self.data_batch['img_metas'].data[0][0]['img_norm_cfg']['std']).view(1, -1, 1, 1).to(self.model.device)

            ori = self.data_batch['img'].data[0].clone().detach().to(self.model.device)

            self.call_hook('before_train_iter')
            self.data_batch['img'].data[0].requires_grad = True
            self.run_iter(data_batch, train_mode=True, **kwargs)

            # losstype['bbox'](self.outputs['unparsedlosses']).backward(retain_graph=True)
            (self.outputs['loss']).backward(retain_graph=True)
            locpert = (ori + FGSMupdater(self.data_batch['img'].data[0].grad.to(self.model.device), ori, mean, std)).clone().detach().cpu()

            # losstype['cls'](self.outputs['unparsedlosses']).backward()
            (self.outputs['loss']).backward()
            clspert = (ori + FGSMupdater(self.data_batch['img'].data[0].grad.to(self.model.device), ori, mean, std)).clone().detach().cpu()

            data_batch['img'].data[0] = locpert
            self.run_iter(data_batch, train_mode=True, **kwargs)
            locloss = self.outputs['loss'].clone().detach()

            data_batch['img'].data[0] = clspert
            self.run_iter(data_batch, train_mode=True, **kwargs)
            clsloss = self.outputs['loss'].clone().detach()

            self.optimizer.zero_grad()
            if clsloss > locloss:
                self.outputs['loss'].backward()
            else:
                data_batch['img'].data[0] = locpert
                self.run_iter(data_batch, train_mode=True, **kwargs)
                self.outputs['loss'].backward()

            self.call_hook('after_train_iter')

            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1