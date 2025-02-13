import torch
import numpy as np

from PIL import Image
import os

# save the patch as numpy
def load_patch(load_patch_name):
    return torch.tensor(np.load(load_patch_name))

def save_patch(patch, dir, epoch):
    patch_size = patch.size()
    patch_np = patch.data.cpu().numpy()

    global save_patch_name

    save_patch_name = os.path.join(dir, '{}.npy'.format(epoch))
    print("save patch as ", save_patch_name)
    np.save(save_patch_name, patch_np)

    patch_img_np = np.zeros((patch_size[-2], patch_size[-1], 3))
    patch_img_np[:, :, 0] = patch_np[0][0]  # B(0)
    patch_img_np[:, :, 1] = patch_np[0][1]  # G(1)
    patch_img_np[:, :, 2] = patch_np[0][2]  # R(2)
    np.transpose(patch_img_np, (2, 1, 0))  # RGB

    patch_img = Image.fromarray(patch_img_np.astype('uint8'))
    save_patch_img = os.path.join(dir, '{}.png'.format(epoch))
    print("save patch as img ", save_patch_img)
    patch_img.save(save_patch_img)