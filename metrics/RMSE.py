import torch
import torch.nn as nn

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

mse = nn.MSELoss()
score = mse(bld_imags, real_imgs)**0.5
print(score)