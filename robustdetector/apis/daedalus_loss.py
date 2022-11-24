import torch
import torch.nn.functional as F

def outputdecode(model, res):
    bbox_pred_img_list = []
    cls_score_img_list = []

    for img_id in range(res[0][0].size(0)):
        cls_score_list = [item[img_id, :] for item in res[0]]
        bbox_pred_list = [item[img_id, :] for item in res[1]]

        bbox_pred_parsed_list = []
        cls_score_parsed_list = []

        for level_idx, (cls_score, bbox_pred) in enumerate(zip(cls_score_list, bbox_pred_list)):
            bbox_pred_parsed_list += bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            cls_score_parsed_list += cls_score.permute(1, 2, 0).reshape(-1, model.module.bbox_head.cls_out_channels)

        bbox_pred_img_list.append(torch.stack(bbox_pred_parsed_list))
        cls_score_img_list.append(torch.stack(cls_score_parsed_list))

    return [torch.stack(bbox_pred_img_list), torch.stack(cls_score_img_list)]

class DaedalusLoss():
    @staticmethod
    def forward(predictions, obj):
        loss_c = torch.zeros((predictions[0].shape[0]), device="cpu")
        loss_l = torch.zeros((predictions[0].shape[0]), device="cpu")

        for xi, (bbox, conf) in enumerate(zip(predictions[0], predictions[1])):  # image index, image inference
            loss_c[xi] = torch.sum(torch.pow((F.softmax(conf, dim=1)[:, -1] - 1), 2))
            loss_l[xi] = torch.sum(bbox[:, 2] * bbox[:, 3])

        if sum(loss_l) > 0 and sum(loss_c) > 0:
            while sum(loss_l) > sum(loss_c):
                loss_l /= 10

            while sum(loss_c) > sum(loss_l):
                loss_c /= 10

        return -sum(loss_c + loss_l / 10)