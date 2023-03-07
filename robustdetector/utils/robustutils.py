import torch

# mean = torch.tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1)

#TODO cast */std
def perturbupdater(perturb, grad, ori, norm_cfg):
    mean = torch.tensor(norm_cfg['mean']).view(1, -1, 1, 1).to(ori.device)
    std = torch.tensor(norm_cfg['std']).view(1, -1, 1, 1).to(ori.device)

    perturb *= std
    perturb += 2 * grad.sign()
    perturb = perturb.clamp(-8, 8)
    perturb = ((ori * std + perturb + mean).clamp(0, 255) - ori * std - mean) / std
    return perturb.detach()