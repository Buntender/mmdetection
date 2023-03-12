import torch

# mean = torch.tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1)

#TODO cast */std
def perturbupdater(perturb, grad, ori, mean, std):
    perturb += 2 * grad.sign()
    perturb = perturb.clamp(-8, 8)
    perturb = ((ori * std + perturb + mean).clamp(0, 255) - ori * std - mean)
    return perturb.detach()

def FGSMupdater(grad, ori, mean, std):
    perturb = 8 * grad.sign()
    perturb = ((ori * std + perturb + mean).clamp(0, 255) - ori * std - mean)
    return perturb.detach()