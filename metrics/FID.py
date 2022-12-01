from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import scipy
from torch.autograd import Variable
import numpy as np

class PartialInception(nn.Module):
    def __init__(self):
        super(PartialInception, self).__init__()
        
        self.inc = models.inception_v3(pretrained=True)
        self.inc.Mixed_7c.register_forward_hook(self.outhook)
    
    def outhook(self, module, inp, output):
        self.mixed_7c_output = output
    
    def forward(self, x):
        self.inc(x)
        act = self.mixed_7c_output
        act = F.adaptive_avg_pool2d(act, (1, 1))
        act = torch.flatten(act, 1)

        return act

model = PartialInception()

def calculate_fid(gen_images, real_images, batch_size):
        real_acts = []
        gen_acts = []
        import math
        step_size = int(math.ceil(len(real_images) / batch_size))
        # print(step_size)
        for i in range(step_size):
            real_batch = real_images[i * batch_size:(i + 1) * batch_size]
            gen_batch = gen_images[i * batch_size:(i + 1) * batch_size]
            print(len(real_batch))
            real_acts.append(Variable(model(real_batch))) # Variable
            gen_acts.append(Variable(model(gen_batch)))
        
        x_features = torch.cat(real_acts, dim=0)
        y_features = torch.cat(gen_acts, dim=0)
        
        score = fid_score(x_features, y_features)
        return score


def covariance(matrix: torch.Tensor, rowvar: bool = True) -> torch.Tensor:

    if matrix.dim() < 2:
        matrix = matrix.view(1, -1)

    if not rowvar and matrix.size(0) != 1:
        matrix = matrix.T

    factor = 1.0 / (matrix.size(1) - 1)
    matrix = matrix - torch.mean(matrix, dim=1, keepdim=True)
    m_t = matrix.T
    return factor * matrix.matmul(m_t).squeeze()


def fid_score(x_features, y_features):
        # GPU -> CPU
        eps=1e-6
        mean_x, sigma_x = torch.mean(x_features, dim=0), covariance(x_features, rowvar=False)
        mean_y, sigma_y = torch.mean(y_features, dim=0), covariance(y_features, rowvar=False)
        mean_diff = mean_x - mean_y
        # covmean, _ = _sqrtm_newton_schulz(sigma_x.mm(sigma_y))
        covmean, _ = scipy.linalg.sqrtm(sigma_x.mm(sigma_y), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        covmean = torch.tensor(covmean)
        # Product might be almost singular
        if not torch.isfinite(covmean).all():
            offset = torch.eye(sigma_x.size(0), device=mean_x.device, dtype=mean_x.dtype) * eps
            # covmean, _ = _sqrtm_newton_schulz((sigma_x + offset).mm(sigma_y + offset))
            covmean, _ = scipy.linalg.sqrtm((sigma_x + offset).dot(sigma_y + offset), disp=False)

        tr_covmean = torch.trace(covmean)
        score = mean_diff.dot(mean_diff) + torch.trace(sigma_x) + torch.trace(sigma_y) - 2 * tr_covmean
    
        return score
    
# You can append as many input and embedded images as you want and calculate the FID score using the code below

import os
from PIL import Image
import torchvision.transforms as transforms

input_images = []
blended_images = []
for im_path in os.listdir('img/input/'):
    im_path = f'img/input/{im_path}'
    print('analyzing: ', im_path)
    inp_img = Image.open(im_path)
    inp_tens = transforms.ToTensor()(inp_img).unsqueeze(0)
    input_images.append(inp_tens[:, :3, :, :])

for im_path in os.listdir('output__/'):
    img_name = im_path
    im_path = f'output__/{im_path}'
    if not img_name.startswith('vis_mask') and '.' in img_name:
        print('analyzing: ', im_path)
        
        bld_img2 = Image.open(im_path)
        bld_img2 = transforms.ToTensor()(bld_img2).unsqueeze(0)
        blended_images.append(bld_img2)
real_imgs = torch.cat(input_images, dim=0)
bld_imags = torch.cat(blended_images, dim=0)

print(real_imgs.shape)
print(bld_imags.shape)

batch_size = len(real_imgs)
print('batch_size', batch_size)
score = calculate_fid(bld_imags, real_imgs, batch_size)
print(score)