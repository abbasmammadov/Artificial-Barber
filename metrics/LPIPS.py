from typing import List, Union, Collection, Optional

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchvision.models import vgg16, vgg19
# Map VGG names to corresponding number in torchvision layer

VGG16_LAYERS = {
    "conv1_1": '0', "relu1_1": '1',
    "conv1_2": '2', "relu1_2": '3',
    "pool1": '4',
    "conv2_1": '5', "relu2_1": '6',
    "conv2_2": '7', "relu2_2": '8',
    "pool2": '9',
    "conv3_1": '10', "relu3_1": '11',
    "conv3_2": '12', "relu3_2": '13',
    "conv3_3": '14', "relu3_3": '15',
    "pool3": '16',
    "conv4_1": '17', "relu4_1": '18',
    "conv4_2": '19', "relu4_2": '20',
    "conv4_3": '21', "relu4_3": '22',
    "pool4": '23',
    "conv5_1": '24', "relu5_1": '25',
    "conv5_2": '26', "relu5_2": '27',
    "conv5_3": '28', "relu5_3": '29',
    "pool5": '30',
}

VGG19_LAYERS = {
    "conv1_1": '0', "relu1_1": '1',
    "conv1_2": '2', "relu1_2": '3',
    "pool1": '4',
    "conv2_1": '5', "relu2_1": '6',
    "conv2_2": '7', "relu2_2": '8',
    "pool2": '9',
    "conv3_1": '10', "relu3_1": '11',
    "conv3_2": '12', "relu3_2": '13',
    "conv3_3": '14', "relu3_3": '15',
    "conv3_4": '16', "relu3_4": '17',
    "pool3": '18',
    "conv4_1": '19', "relu4_1": '20',
    "conv4_2": '21', "relu4_2": '22',
    "conv4_3": '23', "relu4_3": '24',
    "conv4_4": '25', "relu4_4": '26',
    "pool4": '27',
    "conv5_1": '28', "relu5_1": '29',
    "conv5_2": '30', "relu5_2": '31',
    "conv5_3": '32', "relu5_3": '33',
    "conv5_4": '34', "relu5_4": '35',
    "pool5": '36',
}


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Constant used in feature normalization to avoid zero division
EPS = 1e-10
def hann_filter(kernel_size: int) -> torch.Tensor:
    
    # Take bigger window and drop borders
    window = torch.hann_window(kernel_size + 2, periodic=False)[1:-1]
    kernel = window[:, None] * window[None, :]
    # Normalize and reshape kernel
    return kernel.view(1, kernel_size, kernel_size) / kernel.sum()


class L2Pool2d(torch.nn.Module):
    
    EPS = 1e-12

    def __init__(self, kernel_size: int = 3, stride: int = 2, padding=1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel is None:
            C = x.size(1)
            self.kernel = hann_filter(self.kernel_size).repeat((C, 1, 1, 1)).to(x)

        out = torch.nn.functional.conv2d(
            x ** 2, self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1]
        )
        return (out + self.EPS).sqrt()
def _reduce(x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    
    if reduction == 'none':
        return x
    elif reduction == 'mean':
        return x.mean(dim=0)
    elif reduction == 'sum':
        return x.sum(dim=0)
    else:
        raise ValueError("Unknown reduction. Expected one of {'none', 'mean', 'sum'}")
def similarity_map(map_x: torch.Tensor, map_y: torch.Tensor, constant: float, alpha: float = 0.0) -> torch.Tensor:
    r""" Compute similarity_map between two tensors using Dice-like equation.
    Args:
        map_x: Tensor with map to be compared
        map_y: Tensor with map to be compared
        constant: Used for numerical stability
        alpha: Masking coefficient. Subtracts - `alpha` * map_x * map_y from denominator and nominator
    """
    return (2.0 * map_x * map_y - alpha * map_x * map_y + constant) / \
           (map_x ** 2 + map_y ** 2 - alpha * map_x * map_y + constant)


class ContentLoss(_Loss):

    def __init__(self, feature_extractor: Union[str, torch.nn.Module] = "vgg16", layers: Collection[str] = ("relu3_3",),
                 weights: List[Union[float, torch.Tensor]] = [1.], replace_pooling: bool = False,
                 distance: str = "mse", reduction: str = "mean", mean: List[float] = IMAGENET_MEAN,
                 std: List[float] = IMAGENET_STD, normalize_features: bool = False,
                 allow_layers_weights_mismatch: bool = False) -> None:

        assert allow_layers_weights_mismatch or len(layers) == len(weights), \
            f'Lengths of provided layers and weighs mismatch ({len(weights)} weights and {len(layers)} layers), ' \
            f'which will cause incorrect results. Please provide weight for each layer.'

        super().__init__()

        if callable(feature_extractor):
            self.model = feature_extractor
            self.layers = layers
        else:
            if feature_extractor == "vgg16":
                self.model = vgg16(pretrained=True, progress=False).features
                self.layers = [VGG16_LAYERS[l] for l in layers]
            elif feature_extractor == "vgg19":
                self.model = vgg19(pretrained=True, progress=False).features
                self.layers = [VGG19_LAYERS[l] for l in layers]
            else:
                raise ValueError("Unknown feature extractor")

        if replace_pooling:
            self.model = self.replace_pooling(self.model)

        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.distance = {
            "mse": nn.MSELoss,
            "mae": nn.L1Loss,
        }[distance](reduction='none')

        self.weights = [torch.tensor(w) if not isinstance(w, torch.Tensor) else w for w in weights]

        mean = torch.tensor(mean)
        std = torch.tensor(std)
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)

        self.normalize_features = normalize_features
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        
        self.model.to(x)
        x_features = self.get_features(x)
        y_features = self.get_features(y)

        distances = self.compute_distance(x_features, y_features)

        # Scale distances, then average in spatial dimensions, then stack and sum in channels dimension
        loss = torch.cat([(d * w.to(d)).mean(dim=[2, 3]) for d, w in zip(distances, self.weights)], dim=1).sum(dim=1)

        return _reduce(loss, self.reduction)

    def compute_distance(self, x_features: List[torch.Tensor], y_features: List[torch.Tensor]) -> List[torch.Tensor]:
        
        return [self.distance(x, y) for x, y in zip(x_features, y_features)]

    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
    
        # Normalize input
        x = (x - self.mean.to(x)) / self.std.to(x)

        features = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layers:
                features.append(self.normalize(x) if self.normalize_features else x)

        return features

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        r"""Normalize feature maps in channel direction to unit length.
        Args:
            x: Tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Normalized input
        """
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + EPS)

    def replace_pooling(self, module: torch.nn.Module) -> torch.nn.Module:
        
        module_output = module
        if isinstance(module, torch.nn.MaxPool2d):
            module_output = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        for name, child in module.named_children():
            module_output.add_module(name, self.replace_pooling(child))
        return module_output

class LPIPS(ContentLoss):

    _weights_url = "https://github.com/photosynthesis-team/" + \
        "photosynthesis.metrics/releases/download/v0.4.0/lpips_weights.pt"

    def __init__(self, replace_pooling: bool = False, distance: str = "mse", reduction: str = "mean",
                 mean: List[float] = IMAGENET_MEAN, std: List[float] = IMAGENET_STD,) -> None:
        lpips_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        lpips_weights = torch.hub.load_state_dict_from_url(self._weights_url, progress=False)
        super().__init__("vgg16", layers=lpips_layers, weights=lpips_weights,
                         replace_pooling=replace_pooling, distance=distance,
                         reduction=reduction, mean=mean, std=std,
                         normalize_features=True)

# You can append as many input and embedded images as you want 
# and calculate the LPIPS measurement using the code below

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
lpip = LPIPS()
score = lpip(bld_imags, real_imgs)
print(score)