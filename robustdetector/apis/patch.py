import torch
import numpy as np
import os

from PIL import Image

#TODO move to utils
save_patch_path = "work_dirs/ssd300_voc_AdvClock_0303_clsloss_Patch"
if not os.path.isdir(save_patch_path):
    os.makedirs(save_patch_path)

# save the patch as numpy
def load_patch(load_patch_name):
    return torch.tensor(np.load(load_patch_name))

def save_patch(patch, epoch):
    patch_size = patch.size()
    patch_np = patch.data.cpu().numpy()

    global save_patch_name

    save_patch_name = os.path.join(save_patch_path, '{}.npy'.format(epoch))
    print("save patch as ", save_patch_name)
    np.save(save_patch_name, patch_np)

    patch_img_np = np.zeros((patch_size[-2], patch_size[-1], 3))
    patch_img_np[:, :, 0] = patch_np[0][0] * 255.0  # B(0)
    patch_img_np[:, :, 1] = patch_np[0][1] * 255.0  # G(1)
    patch_img_np[:, :, 2] = patch_np[0][2] * 255.0  # R(2)
    np.transpose(patch_img_np, (2, 1, 0))  # RGB

    patch_img = Image.fromarray(patch_img_np.astype('uint8'))
    save_patch_img = os.path.join(save_patch_path, '{}.png'.format(epoch))
    print("save patch as img ", save_patch_img)
    patch_img.save(save_patch_img)