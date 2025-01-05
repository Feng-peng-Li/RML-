import torch
import torch.nn as nn
import torchvision.transforms

import random
import numpy as np
import torch.nn.functional as F
import math
from torchvision import transforms
from PIL import Image

from torchvision.utils import save_image



def mask(sample_x, maps, theta):
        batches, ch, imgH, imgW = sample_x.size()
        maps_resize = maps.view(batches, 1, imgH * imgW)
        k = imgH * imgW * theta
        thres, _ = torch.topk(maps_resize, k=int(k), dim=2, largest=True)
        thres,_ = thres.min(dim=2)
        thres = thres.unsqueeze(2).repeat(1, 1, imgH * imgW).view(batches, 1, imgH, imgW)
        mask = (maps - thres) >=0.
        # masked_img = mask * sample_x

        return mask

def core_extraction(adv_p,sample_x,theta,padding_ratio=0.1):
    core_sample=[]
    batch,ch,w,h=sample_x.size()
    # adv_p=torch.abs(adv_p)
    adv_p=torch.mean(adv_p,dim=1,keepdim=True)
    for i in range(batch):
        sample=sample_x[i:i+1]
       
        adv_map = adv_p[i:i + 1]

        k = h * w * theta
        adv_map_r = adv_map.view(1, 1, h * w)
        thres, _ = torch.topk(adv_map_r, k=int(k), dim=2, largest=True)
        thres, _ = thres.min(dim=2)
        adv_mask = adv_map >=thres
        mask = adv_mask 
        core_sample.append(up_sample(mask, sample, padding_ratio=0.1))
    core_sample = torch.cat(core_sample, dim=0)
    return core_sample


def up_sample(mask,sample,padding_ratio):

        nonzero_indices = torch.nonzero(mask[0, 0, ...])
        _, _, imgH, imgW = sample.size()
        if nonzero_indices.numel() != 0:

            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
            crop_single_heat = F.upsample_bilinear(
                sample[ :, :, height_min:height_max, width_min:width_max],
                size=(imgH, imgW))
            return crop_single_heat
        else:
            return sample


