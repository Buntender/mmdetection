import torch
import torch.nn.functional as F
from itertools import chain

def outputdecode(model, res):
    cls_score_img_list = [layer.permute(0, 2, 3, 1).reshape(layer.size(0), -1, model.module.bbox_head.cls_out_channels)
                          for layer in res[0]]
    bbox_pred_img_list = [layer.permute(0, 2, 3, 1).reshape(layer.size(0), -1, 4)
                          for layer in res[1]]

    return [torch.cat(cls_score_img_list, dim=1), torch.cat(bbox_pred_img_list, dim=1)]

class DaedalusLoss():
    bboxes = [5776, 2166, 600, 150, 36, 4]
    defaultbox = [30, 60, 111, 162, 213, 264]
    imgsize = 300

    def __init__(self):
        deafaultsize = [pow(item / self.imgsize, 2) for item in self.defaultbox]
        self.weights = [[deafaultsize[item]] * self.bboxes[item] for item in range(len(self.bboxes))]
        self.weights = torch.tensor(list(chain(*self.weights)))

    def forward(self, predictions, obj):
        loss_c = torch.mean(torch.pow((F.softmax(predictions[0], dim=1)[:, -1] - 1), 2))
        loss_l = torch.exp(2 * (predictions[1][:,:,-1] + predictions[1][:,:,-2]))
        self.weights = self.weights.to(loss_l.device)

        return loss_c + torch.mean(loss_l * self.weights)