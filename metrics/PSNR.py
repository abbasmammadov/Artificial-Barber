import torch

def psnr(x, y, data_range = (0, 1), reduction: str = 'mean', to_greyscale = False):
    # Constant for numerical stability
    EPS = 1e-8
    values_range = data_range[1] - data_range[0]
    x = x / values_range
    y = y / values_range

    if (x.size(1) == 3) and to_greyscale:
        # Convert RGB image to YIQ and take luminance: Y = 0.299 R + 0.587 G + 0.114 B
        rgb_to_grey = torch.tensor([0.299, 0.587, 0.114]).view(1, -1, 1, 1).to(x)
        x = torch.sum(x * rgb_to_grey, dim=1, keepdim=True)
        y = torch.sum(y * rgb_to_grey, dim=1, keepdim=True)

    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    score = - 10 * torch.log10(mse + EPS)

    if reduction == 'sum':
        score = score.sum(dim=0)
    else:
        score = score.mean(dim=0)
   
    return score

# You can append as many input and embedded images as you want 
# and calculate the PSNR score using the code below
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
score = psnr(bld_imags, real_imgs, reduction='mean')
print(score)