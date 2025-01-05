import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Generator_MLP(nn.Module):
    def __init__(self, in_channel=3*32*32, out_channel=1, img_h=32, img_w=32,ksize=5):
        super(Generator_MLP, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.h = img_h
        self.w = img_w
        self.ksize=ksize

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.in_channel, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, self.out_channel * self.h * self.w),

        )
        self.apply(init_weights)

    def tensor_dilate(self,bin_img):  #
        # 首先为原图加入 padding，防止图像尺寸缩小
        B, C, H, W = bin_img.shape
        pad = (self.ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
        # 将原图 unfold 成 patch
        patches = bin_img.unfold(dimension=2, size=self.ksize, step=1)
        patches = patches.unfold(dimension=3, size=self.ksize, step=1)
        # B x C x H x W x k x k
        # 取每个 patch 中最小的值，i.e., 0
        dilate, _ = patches.reshape(B, C, H, W, -1).median(dim=-1)
        return dilate
    def forward(self, perturb,sample):
        perturb=perturb.view(perturb.size(0),-1)
        mask = self.model(perturb)
        mask = mask.view(mask.shape[0], self.out_channel, self.h, self.w)
        mask=mask.repeat(1,sample.size(1),1,1)
        mask=torch.where(mask>0,1,0)

        p_img=sample*mask
        p_img=self.tensor_dilate(p_img)

        mask_index=torch.where(mask==0)
        sample[mask_index]=p_img[mask_index]



        return sample