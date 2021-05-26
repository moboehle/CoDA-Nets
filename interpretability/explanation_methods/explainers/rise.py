#
#     Adpated from https://github.com/eclique/RISE/blob/master/explanations.py
#
import os

import numpy as np
import torch
from skimage.transform import resize
from torch import nn as nn
from tqdm import tqdm

from interpretability.explanation_methods.utils import ExplainerBase, limit_n_images
from project_config import RISE_MASK_PATH


class RISE(ExplainerBase, nn.Module):
    # for each image size, create masks only once and save them as masks{HEIGHT}.py
    # This assumes square images..
    path_tmplt = os.path.join(RISE_MASK_PATH, "masks{}.npy")

    def __init__(self, trainer, batch_size=2, n=6000, s=6, p1=.1):
        ExplainerBase.__init__(self, trainer)
        nn.Module.__init__(self)
        self.batch_size = batch_size
        self.max_imgs_bs = 1
        self.N = n
        self.s = s
        self.p1 = p1
        self.masks = None

    def generate_masks(self, savepath='masks.npy', input_size=None):
        print("Generating masks for", input_size)
        p1, s = self.p1, self.s
        if not os.path.isdir(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        cell_size = np.ceil(np.array(input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(self.N, s, s) < p1
        grid = grid.astype('float32')

        masks = np.empty((self.N, *input_size))

        for i in tqdm(range(self.N)):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
        masks = masks.reshape(-1, 1, *input_size)
        np.save(savepath, masks)

    def load_masks(self, filepath):
        if not os.path.exists(filepath):
            size = int(os.path.basename(filepath)[len("masks"):-len(".npy")])
            self.generate_masks(savepath=filepath[:-4], input_size=(size, size))
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]
        return self.masks

    @limit_n_images
    @torch.no_grad()
    def attribute(self, x, target, return_all=False):
        N = self.N
        _, _, H, W = x.size()
        if self.masks is None or self.masks.shape[-1] != H:
            self.masks = self.load_masks(self.path_tmplt.format(int(H)))
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)

        p = []
        for i in range(0, N, self.batch_size):
            p.append((self.trainer.predict(stack[i:min(i + self.batch_size, N)])))
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, 1, H, W))
        sal = sal / N / self.p1
        if return_all:
            return sal
        return sal[int(target)][None]

    def attribute_selection(self, x, tgts):
        return self.attribute(x, tgts, return_all=True)[tgts]
