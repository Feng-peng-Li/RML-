from __future__ import print_function
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import Bar, Logger, AverageMeter, accuracy
from utils_awp import TradesAWP
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, Subset
from PIL import Image
from generator import Generator_MLP
import random

# import jax.numpy as jnp
# from autoaugment import CIFAR10Policy
from torch.optim.lr_scheduler import MultiStepLR
from models.resnet import ResNet18
import models

import torchvision
import kornia.augmentation as K

import random
from torchvision.utils import save_image

from extraction_core import core_extraction
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode as Interpolation

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--arch', type=str, default='ResNet18')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--js_weight', default=2., type=int, metavar='N',
                    help='The weight of the JS divergence term')
parser.add_argument('--mask_t', default=0.1, type=int,
                    help='mask_t')
parser.add_argument('--loss', default='mse', type=str,
                    help='perturbation', choices=['mse', 'M', 'cosine'])
parser.add_argument('--c_epoch', type=int, default=50, metavar='N',
                    help='retrain from which epoch')
parser.add_argument('--gpu_id', default=0, type=int,
                    help='gpu')
parser.add_argument('--ksize', type=int, default=3, metavar='ksize',
                    help='ksize')
parser.add_argument('--p_size', type=int, default=4, metavar='psize',
                    help='psize')
parser.add_argument('--beta', default=6, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--w', default=50.00, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--lam', type=int, default=0.9, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--lam_alpha', type=int, default=0.4
                    , metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--step2', default=110, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                    help='retrain from which epoch')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--in_ch', default=512, type=int,
                    help='channel of input noise')

parser.add_argument('--alpha', type=float, default=0, metavar='mix_beta1',
                    help='mix_rate')
parser.add_argument('--mix_num', default=10, type=int,
                    help='perturbation')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train')

parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
parser.add_argument('--data-path', type=str, default='./data',
                    help='where is the dataset CIFAR-10')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'],
                    help='The threat model')
parser.add_argument('--train_budget', default='high', type=str, choices=['low', 'high'],
                    help='The compute budget for training. High budget would mean larger number of atatck iterations')
parser.add_argument('--epsilon', default=8, type=float,
                    help='perturbation')
parser.add_argument('--step-size', default=2, type=float,
                    help='perturb step size')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--LS', type=int, default=0, metavar='S',
                    help='make 1 is want to use Label Smoothing. DAJAT uses LS only for CIFAR10 dataset')
parser.add_argument('--model-dir', default='./model-cifar-ResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--resume-model', default='/home/yc17420/trades_AWP/model-cifar-ResNet_0_10/ours-model-epoch40.pt',
                    type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim',
                    default='/home/yc17420/trades_AWP/model-cifar-ResNet_0_10/ours-opt-checkpoint_epoch40.tar',
                    type=str,
                    help='directory of optimizer for retraining')
parser.add_argument('--resume-model1',
                    default='', type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim1',
                    default='',
                    type=str,
                    help='directory of optimizer for retraining')
parser.add_argument('--resume-model2',
                    default='', type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim2',
                    default='',
                    type=str,
                    help='directory of optimizer for retraining')
parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--num_auto', default=1, type=int, metavar='N',
                    help='Number of autoaugments to use for training')

parser.add_argument('--awp-gamma', default=0.005, type=float,
                    help='whether or not to add parametric noise')
parser.add_argument('--awp-warmup', default=10000, type=int,
                    help='We could apply AWP after some epochs for accelerating.')
parser.add_argument('--use_defaults', type=str, default='NONE',
                    choices=['NONE', 'CIFAR10_RN18', 'CIFAR10_WRN', 'CIFAR100_WRN', 'CIFAR100_RN18'],
                    help='Use None is want to use the hyperparamters passed in the python training command else use the desired set of default hyperparameters')

args = parser.parse_args()
# if args.use_defaults != 'NONE':
#     args = use_default(args.use_defaults)
print(args)

epsilon = args.epsilon / 255
args.epsilon = epsilon
if args.awp_gamma <= 0.0:
    args.awp_warmup = np.infty
if args.data == 'CIFAR100':
    NUM_CLASSES = 100
else:
    NUM_CLASSES = 10
    idbh_p = 'cifar10-weak'

# settings
model_dir = args.model_dir + '_' + str(args.gpu_id) + '_' + str(args.mix_num)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.gpu_id)
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}


class Get_Dataset_C10(torchvision.datasets.CIFAR10):

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        image = Image.fromarray(img)
        image_clean = self.transform(image)
        # supp_list = []
        # for i in range(100000):
        #     pair_index = random.randint(0, len(self.data) - 1)
        #     if pair_index != idx and self.targets[pair_index]==target:
        #         supp_list.append(pair_index)
        #     if len(supp_list) >= args.mix_num:
        #         break
        # supp_data = []
        # for i in supp_list:
        #     supp_data.append(self.transform(Image.fromarray(self.data[i])))

        return image_clean, target, idx


class SemanticMI(torch.nn.Module):
    def __init__(self, num_classes, dim_hidden, temperature=0.5, master_rank="cuda", DDP=False):
        super(SemanticMI, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.master_rank = master_rank
        self.DDP = DDP
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, embed, proxy, label, **_):
        # print(embed.size(),proxy.size())
        sim_matrix_e2p = self.calculate_similarity_matrix(embed, proxy)
        sim_matrix_e2p = torch.exp(self._remove_diag(sim_matrix_e2p) / self.temperature)
        neg_removal_mask = self._remove_diag(self._make_neg_removal_mask(label)[label])
        pos_removal_mask = 1 - neg_removal_mask
        sim_neg_only_e2p = pos_removal_mask * sim_matrix_e2p
        loss_pos = -torch.log(1 - (sim_neg_only_e2p / (sim_neg_only_e2p + 0.01))).mean()
        emb2proxy = torch.exp(self.cosine_similarity(embed, proxy) / self.temperature)
        loss_neg = -torch.log(emb2proxy / (emb2proxy + 0.01)).mean()

        return loss_pos + loss_neg


# class ContrastiveCenterLoss(nn.Module):
#     def __init__(self, dim_hidden, num_classes, lambda_c=1.0, use_cuda=True) -> object:
#         super(ContrastiveCenterLoss, self).__init__()
#         self.dim_hidden = dim_hidden
#         self.num_classes = num_classes
#         self.lambda_c = lambda_c
#         self.use_cuda = use_cuda
#
#     # may not work due to flowing gradient. change center calculation to exp moving avg may work.
#     def forward(self, hidden, feature_center, y, reduction=True):
#         batch_size = hidden.size()[0]
#         one_y = F.one_hot(y, self.num_classes)
#         # feature_c = F.normalize(feature_center,dim=1)
#         # feature_h = F.normalize(hidden,dim=1)
#
#         feature_c = feature_center.unsqueeze(0).repeat(batch_size, 1, 1)
#         feature_h = hidden.unsqueeze(1).repeat(1, self.num_classes, 1)
#
#         feature_c = F.normalize(feature_c, dim=2)
#         feature_h = F.normalize(feature_h, dim=2)
#
#         dis = torch.sum((feature_h - feature_c) ** 2, dim=2)
#
#         # intra_dis=F.softmax(-dis,dim=1)
#         # inter_dis=F.softmax(dis,dim=1)
#         #
#         # intra_p = torch.sum(intra_dis * one_y, dim=1)
#         # inter_p = torch.sum(inter_dis * (1 - one_y), dim=1)
#         # loss=-torch.log(intra_p)-self.lambda_c*torch.log(inter_p)
#
#         intra_p = torch.sum(dis * one_y, dim=1)
#         inter_p = torch.sum(dis * (1 - one_y), dim=1) / (NUM_CLASSES - 1)
#         loss = intra_p +self.lambda_c * (-inter_p)
#
#         if reduction:
#             loss = loss.mean()
#         return loss

class ContrastiveCenterLoss(nn.Module):
    def __init__(self, dim_hidden, num_classes, lam=0.9, lambda_c=1.0, use_cuda=True):
        super(ContrastiveCenterLoss, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c

        self.use_cuda = use_cuda

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, hidden, centers, y):
        batch_size = hidden.size()[0]

        expanded_centers = centers.expand(batch_size, -1, -1)
        expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1, 0)
        distance_centers=(expanded_hidden-expanded_centers)**2
        distance_centers=torch.sum(distance_centers,dim=2)

        one_y=F.one_hot(y,NUM_CLASSES)

        intra_distances=torch.sum(distance_centers*one_y,dim=1)
        inter_distances=torch.sum(distance_centers*(1-one_y),dim=1)


        # distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1)
        # distances_same = distance_centers.gather(1, y.unsqueeze(1))
        # intra_distances = distances_same.sum()
        # inter_distances = distance_centers.sum().sub(intra_distances)
        epsilon = 1e-6
        # loss = (self.lambda_c / 2.) * intra_distances /(inter_distances + epsilon) / 0.1
        # loss=(intra_distances-self.lambda_c*inter_distances/(NUM_CLASSES-1))/batch_size
        loss=intra_distances/(inter_distances/(NUM_CLASSES-1)+epsilon)
        return loss.mean()


# class ContrastiveCenterLoss(nn.Module):
#     def __init__(self, dim_hidden,num_classes,lambda_c=2.0, use_cuda=True):
#         super(ContrastiveCenterLoss, self).__init__()
#         self.dim_hidden = dim_hidden
#         self.num_classes = num_classes
#         self.lambda_c = lambda_c
#         self.m=0.2
#         self.use_cuda = use_cuda
#         self.temperature=0.02
#     # may not work due to flowing gradient. change center calculation to exp moving avg may work.
#     def forward(self,hidden,centers,y):
#         batch_size = hidden.size()[0]
#         centers=F.normalize(centers,dim=1)
#         hidden=F.normalize(hidden,dim=1)
#
#         one_y=F.one_hot(y,NUM_CLASSES)
#         centers=centers.unsqueeze(0).repeat(batch_size,1,1)
#         hidden=hidden.unsqueeze(1).repeat(1,NUM_CLASSES,1)
#
#         all_dis=torch.sum(centers*hidden,dim=2)
#
#         pos_dis=torch.sum(all_dis*one_y,dim=1)
#         neg_dis=torch.sum(torch.exp(all_dis/self.temperature)*(1-one_y),dim=1)
#
#         com_dis=torch.exp((pos_dis-self.m)/self.temperature+2*self.lambda_c*pos_dis)/(torch.exp((pos_dis-self.m)/self.temperature)+neg_dis+1e-6)
#
#         loss=-torch.log(com_dis)
#
#         return loss.mean()


class ContrastiveCenterLoss_new(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c=1.0, use_cuda=True):
        super(ContrastiveCenterLoss_new, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.use_cuda = use_cuda

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, hidden, feature_center, y, reduction=True):
        batch_size = hidden.size()[0]
        one_y = F.one_hot(y, self.num_classes)
        # intra_distance=F.cosine_similarity(hidden,feature_center[y],dim=1)
        # intra_distance=torch.mean(intra_distance,dim=0)

        center = feature_center.unsqueeze(0).repeat(batch_size, 1, 1)
        c_feature = hidden.unsqueeze(1).repeat(1, self.num_classes, 1)
        # center = F.normalize(center,dim=2)
        # c_feature = F.normalize(c_feature,dim=2)

        dis = F.cosine_similarity(c_feature, center, dim=2)

        intra_distance = torch.sum(dis * one_y, dim=1)
        inter_distance = torch.sum(dis * (1 - one_y), dim=1) / (NUM_CLASSES - 1)

        loss = (1 - intra_distance) + self.lambda_c * inter_distance
        if reduction:
            loss = loss.mean()
        # print(inter_distance)
        # print(intra_distance)
        return loss


class ContrastiveCenterLoss_M(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c=1.0, use_cuda=True):
        super(ContrastiveCenterLoss_M, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.use_cuda = use_cuda

    def batch_cov(self, points):
        B, N, D = points.size()
        mean = points.mean(dim=1).unsqueeze(1)
        diffs = (points - mean).reshape(B * N, D)
        prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
        bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
        return bcov

    def Maha_distance(self, X, Y):
        bs, a, b = X.size()
        X = X.view(bs, a, 8, int(b / 8)).mean(dim=3)
        Y = Y.view(bs, a, 8, int(b / 8)).mean(dim=3)
        X = X / (torch.norm(X, dim=2, keepdim=True) + 1e-6)
        Y = Y / (torch.norm(Y, dim=2, keepdim=True) + 1e-6)

        XY = torch.cat([X, Y], dim=1)
        bcov = self.batch_cov(XY)

        X_Y = X - Y

        # pinv_cov=np.linalg.pinv(bcov.detach().cpu().numpy())
        # cov_tmp=torch.linalg.inv(torch.bmm(bcov,bcov.permute(0,2,1)))
        # P=torch.linalg.inv(torch.linalg.cholesky(bcov))
        # pinv_cov=torch.bmm(P.permute(0,2,1),P)
        pinv_cov = torch.linalg.pinv(bcov, hermitian=True)
        # pinv_cov=torch.bmm(cov_tmp,bcov)

        m_dis = torch.bmm(X_Y, pinv_cov)
        m_dis = torch.linalg.diagonal(torch.bmm(m_dis, X_Y.permute(0, 2, 1)))

        return torch.sqrt(m_dis)

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, hidden, feature_center, y, reduction=True):
        batch_size = hidden.size()[0]
        one_y = F.one_hot(y, self.num_classes)

        center = feature_center.unsqueeze(0).repeat(batch_size, 1, 1)
        c_feature = hidden.unsqueeze(1).repeat(1, self.num_classes, 1)
        # center=center/(torch.norm(center,dim=2,keepdim=True)+1e-6)
        # c_feature = c_feature / (torch.norm(c_feature, dim=2, keepdim=True) + 1e-6)
        dis = self.Maha_distance(c_feature, center)
        intra_maha = torch.sum(dis * one_y, dim=1)
        inter_maha = torch.sum(dis * (1 - one_y), dim=1) / (NUM_CLASSES - 1)

        intra_loss = intra_maha
        inter_loss = -inter_maha
        loss = intra_loss + self.lambda_c * inter_loss
        if reduction:
            loss = loss.mean()

        # print(inter_distance)
        # print(intra_distance)
        return loss


class Get_Dataset_C100(torchvision.datasets.CIFAR100):

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        image = Image.fromarray(img)
        image_clean = self.transform[0](image)
        image_auto1 = self.transform[1](image)
        image_auto2 = self.transform[1](image)
        return image_clean, image_auto1, image_auto2, target


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),

])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.data == 'CIFAR10':
    trainset = Get_Dataset_C10(root=args.data_path, train=True, transform=transform_train,
                               download=True)

elif args.data == 'CIFAR100':
    trainset = Get_Dataset_C100(root=args.data_path, train=True, transform=transform_train,
                                download=True)

# class_inds = [torch.where(trainset.targets == class_idx)[0]
#               for class_idx in range(NUM_CLASSES)]


testset = getattr(datasets, args.data)(root=args.data_path, train=False, download=True, transform=transform_test)
# valset = getattr(datasets, args.data)(root=args.data_path, train=True, download=True, transform=transform_test)

# train_size = 49000
# valid_size = 1000
# test_size  = 10000
# train_indices = list(range(50000))
# val_indices = []
# count = np.zeros(100)
# for index in range(len(trainset)):
#     _,_,_, target = trainset[index]
#     if(np.all(count==10)):
#         break
#     if(count[target]<10):
#         count[target] += 1
#         val_indices.append(index)
#         train_indices.remove(index)


train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

# val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,sampler=SubsetRandomSampler(val_indices), **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def perturb_input(model,
                  x_natural,
                  class_center,
                  target,
                  c_center,
                  epoch,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf',
                  PGD_attack=False):
    model.eval()
    batch_size = len(x_natural)
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        # x_adv = x_natural.detach() + (
        #             (4.0 / 255.0) * torch.sign(torch.tensor([0.5]).to(device) - torch.rand_like(x_natural)).to(device))
        for _ in range(perturb_steps):
            x_adv.requires_grad_()

            with torch.enable_grad():
                logits_adv, feature_f = model(x_adv, if_f=True)

                # loss_kl1 = F.kl_div(F.log_softmax(logits_adv, dim=1),
                #                        F.softmax(model(x_natural), dim=1),
                #                        reduction='batchmean')+args.w*c_center*sigmoid_rampup(epoch,1,100)*(feature_f,class_center,target)
                if args.in_ch == 512:
                    loss_contrast = args.w * sigmoid_rampup(epoch, args.c_epoch, 105) * c_center(
                        F.log_softmax(feature_f,dim=1), class_center, target)
                else:
                    loss_contrast = args.w * sigmoid_rampup(epoch, args.c_epoch, args.step2) * c_center(
                        F.log_softmax(logits_adv, dim=1), class_center, target)
                loss_kl1 = F.cross_entropy(logits_adv, target, label_smoothing=args.lam_alpha) + loss_contrast

            grad_na = torch.autograd.grad(loss_kl1, [x_adv])

            x_adv = x_adv.detach() + step_size * torch.sign(grad_na[0].detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            if _==0:
                x_sam=x_adv
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        if PGD_attack:

            x_adv_pgd = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
            for _ in range(perturb_steps):
                x_adv_pgd.requires_grad_()
                with torch.enable_grad():
                    loss_pgd = F.cross_entropy(model(x_adv_pgd), target)
                grad = torch.autograd.grad(loss_pgd, [x_adv_pgd])[0]
                x_adv_pgd = x_adv_pgd.detach() + step_size * torch.sign(grad.detach())
                x_adv_pgd = torch.min(torch.max(x_adv_pgd, x_natural - epsilon), x_natural + epsilon)
                x_adv_pgd = torch.clamp(x_adv_pgd, 0.0, 1.0)


    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * F.kl_div(F.log_softmax(model(adv), dim=1),
                                       F.softmax(model(x_natural), dim=1),
                                       reduction='sum')
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            # if (grad_norms == 0).any():
            #     delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    if PGD_attack:
        return x_adv, x_adv_pgd
    else:
        return x_adv,x_sam




def obtain_center(Memotry_bank, target_bank, selected_num, current_target):
    selected_bank = []
    for target in range(NUM_CLASSES):
        t_index = torch.nonzero(target_bank == target)
        s_index = np.random.randint(t_index.size(0), selected_num)
        selected_M = Memotry_bank[s_index]
        selected_M = torch.mean(selected_M, dim=0, keepdim=True)
        selected_bank.append(selected_M)
    selected_bank = torch.cat(selected_bank, dim=0)

    return selected_bank





def adv_train(model, c_criterion, class_center, train_loader, optimizer,epoch, awp_adversary, exp_avgs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    step_size = 2. / 255
    epsilon = 8. / 255
    num_steps = 10
    print('epoch: {}'.format(epoch))
    bar = Bar('Processing', max=len(train_loader))

    K_aug = nn.Sequential(
        K.RandomHorizontalFlip(),
        K.RandomCrop((32, 32), cropping_mode='resample', padding=1),

    )
    feature_all = []
    target_all = []
    feature_core_all=[]
    # s_it=iter(supp_loader)

    for batch_idx, (data, target, index) in enumerate(train_loader):

        x_natural, target = data.to(device), target.to(device)


        target_all.append(target.detach())

        x_adv,x_sam = perturb_input(model=model,
                              x_natural=x_natural,
                              class_center=class_center,
                              target=target,
                              epoch=epoch,
                              c_center=c_criterion,
                              step_size=step_size,
                              epsilon=epsilon,
                              perturb_steps=num_steps,
                              distance=args.norm,
                              PGD_attack=False)

        # x_perturb = x_adv - x_natural
        # x_core= core_extraction(x_perturb, x_natural, theta=args.mask_t)
        #
        #
        # # x_core,x_core_perturb = adv_sam(x_perturb, x_natural, theta=args.mask_t,ksize=args.ksize)
        # # x_adv_core=x_core+x_core_perturb
        # # x_adv_core=torch.min(torch.max(x_adv_core, x_core - epsilon), x_core + epsilon)
        # # x_adv_core = torch.clamp(x_adv_core, 0.0, 1.0)
        # x_core = K_aug(x_core)
        # x_adv_core,_ = perturb_input(model=model,
        #                            x_natural=x_core,
        #                            class_center=class_center,
        #                            target=target,
        #                            epoch=epoch,
        #                            c_center=c_criterion,
        #                            step_size=step_size,
        #                            epsilon=epsilon,
        #                            perturb_steps=num_steps,
        #                            distance=args.norm,
        #                            PGD_attack=False)


        model.train()

        if epoch >= args.awp_warmup:
            awp = awp_adversary.calc_awp(inputs_adv=x_adv, inputs_clean=x_natural, targets=target, beta=args.beta,
                                         smoothing=args.lam_alpha)
            awp_adversary.perturb(awp)
        optimizer.zero_grad()
        logits_adv_na, adv_na_feature = model(x_adv, if_f=True)
        logits_na, feature_na = model(x_natural, if_f=True)
        if args.in_ch == 512:
            feature_all.append(F.log_softmax(feature_na.detach(),dim=1))
            # feature_all.append(F.log_softmax(adv_na_feature.detach(),dim=1))
        else:
            feature_all.append(F.log_softmax(logits_na.detach(),dim=1))


        loss_robust_na = F.kl_div(F.log_softmax(logits_adv_na, dim=1),
                                  F.softmax(logits_na, dim=1),
                                  reduction='batchmean')
        # loss_robust_na = F.mse_loss(logits_adv_na,logits_na)
        # loss_natural=-torch.sum(torch.log_softmax(logits_na,dim=1)*mix_target,dim=1).mean()
        loss_natural = F.cross_entropy(logits_na, target, label_smoothing=args.lam_alpha)

        # logits_adv_core, adv_core_feature = model(x_adv_core, if_f=True)
        # logits_core, core_feature = model(x_core, if_f=True)
        # if args.in_ch == 512:
        #     feature_core_all.append(core_feature.detach())
        # else:
        #     feature_core_all.append(logits_core.detach())
        #
        # loss_core = F.cross_entropy(logits_core, target, label_smoothing=args.lam_alpha)
        # loss_robust_core = F.kl_div(F.log_softmax(logits_adv_core, dim=1),
        #                             F.softmax(logits_core, dim=1),
        #                             reduction='batchmean')
        # # loss_robust_core = F.mse_loss(logits_adv_core,
        # #                             logits_core)
        # p_mix = torch.clamp(
        #     (F.softmax(logits_core, dim=1) + F.softmax(logits_na, dim=1)) / 2., 1e-7,
        #     1).log()
        # loss_JS = (F.kl_div(p_mix, F.softmax(logits_na, dim=1), reduction='batchmean') + F.kl_div(p_mix,
        #                                                                                           F.softmax(logits_core,
        #                                                                                                     dim=1),
        #                                                                                           reduction='batchmean')) / 2.
        # +args.js_weight * loss_JS
        # loss_JS=symmkl(logits_na,logits_core)
        if args.in_ch == 512:
            # loss_contrast = args.w * sigmoid_rampup(epoch, args.c_epoch, args.step2) * (c_criterion(
            #     F.log_softmax(adv_na_feature,dim=1), class_center, target) + c_criterion(
            #     F.log_softmax(adv_core_feature,dim=1),class_center, target)) / 2.
            loss_contrast = args.w * sigmoid_rampup(epoch, args.c_epoch, args.step2) * (c_criterion(
                F.log_softmax(adv_na_feature, dim=1), class_center, target) )
        else:
            loss_contrast = args.w * sigmoid_rampup(epoch, args.c_epoch, args.step2) * (c_criterion(
                F.log_softmax(logits_adv_na,dim=1), class_center, target) + c_criterion(
                F.log_softmax(logits_adv_core,dim=1), class_center, target)) / 2.
        # loss = (loss_natural + loss_core + args.beta * loss_robust_core + args.beta * loss_robust_na) / 2.+ loss_contrast+args.js_weight*loss_JS
        loss=loss_natural + args.beta * loss_robust_na+loss_contrast

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)


        prec1, prec5 = accuracy(logits_adv_na, target, topk=(1, 5))
        losses.update(loss.item(), x_natural.size(0))
        top1.update(prec1.item(), x_natural.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total:{total:}| ETA:{eta:}| Loss:{loss:.4f}| top1:{top1:.2f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg)
        bar.next()

    feature_all = torch.cat(feature_all, dim=0)
    target_all = torch.cat(target_all, dim=0)

    if epoch >= 0:
        for i in range(NUM_CLASSES):
            index_i = torch.nonzero(target_all == i)
            # print(index_i)
            feature_i = feature_all[index_i]
            feature_i = torch.mean(feature_i, dim=0, keepdim=True)
            feature_i = F.log_softmax(feature_i,dim=1)

            class_center[i:i+1] = (1-args.lam)* feature_i+ args.lam *class_center[i:i + 1]
            # class_center[i:i+1]=(class_center[i:i+1]*epoch+class_center_i)/(epoch+1)
    class_center=F.log_softmax(class_center,dim=1)
    # class_center = class_center/(torch.sum(class_center,dim=1,keepdim=True)+1e-6)

    bar.finish()
    return losses.avg, top1.avg, exp_avgs, class_center


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


upper_limit, lower_limit = 1, 0


def attack_pgd_train(model, X, y, epsilon, alpha, attack_iters, restarts,
                     norm, c_creterion, class_center=None, epoch=0, early_stop=False,
                     mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output, feature = model(X + delta, if_f=True)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(X + delta), y_a, y_b, lam)
            else:

                loss = F.cross_entropy(output, y, label_smoothing=args.lam_alpha) \
                       + args.w * sigmoid_rampup(epoch, args.c_epoch, args.step2) * c_creterion(feature, class_center,
                                                                                                y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(X + delta), y_a, y_b, lam)
        else:
            output, feature = model(X + delta, if_f=True)
            all_loss = F.cross_entropy(output, y, label_smoothing=args.lam_alpha, reduction='none') \
                       + args.w * sigmoid_rampup(epoch, args.c_epoch, args.step2) * c_creterion(feature, class_center,
                                                                                                y, reduction=False)
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, soft_label=None, epoch=0, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(X + delta), y_a, y_b, lam)
            else:

                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(X + delta), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(X + delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def test(model, test_loader, criterion):
    global best_acc
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(test_loader))

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        x_perturb = attack_pgd(model, inputs, targets, 8 / 255., 2 / 255., 20, 1, args.norm)
        x_perturb.detach()
        x_adv = torch.clamp(inputs + x_perturb[:inputs.size(0)], min=0., max=1.0)
        outputs = model(x_adv)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total: {total:}| ETA: {eta:}| Loss:{loss:.4f}| top1: {top1:.2f}'.format(
            batch=batch_idx + 1,
            size=len(test_loader),
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg)
        bar.next()
    bar.finish()
    return losses.avg, top1.avg


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.1
    if epoch >= 105:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def sigmoid_rampup(current, start_es, end_es):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    import math
    if current < start_es:
        return 0

    if current > end_es:
        return 1.0
    else:
        phase = 1.0 - (current - start_es) / (end_es - start_es)
        return math.exp(-5.0 * phase * phase)


def main():
    ################### Change here to WideResNet34 if you want to train on WRN-34-10 #############################
    model = getattr(models, args.arch)(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ################### Change here to WideResNet34 if you want to train on WRN-34-10 #############################
    proxy = getattr(models, args.arch)(num_classes=NUM_CLASSES).to(device)
    proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
    awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.alpha)
    # c_criterion=SemanticMI(num_classes=NUM_CLASSES,dim_hidden=512)
    if args.in_ch == NUM_CLASSES:
        hidden_dim = NUM_CLASSES
    else:
        hidden_dim = args.in_ch

    if args.loss == 'M':
        c_criterion = ContrastiveCenterLoss_M(dim_hidden=hidden_dim, num_classes=NUM_CLASSES)
    if args.loss == 'cosine':
        c_criterion = ContrastiveCenterLoss_new(dim_hidden=hidden_dim, num_classes=NUM_CLASSES)
    if args.loss == 'mse':
        c_criterion = ContrastiveCenterLoss(dim_hidden=hidden_dim, num_classes=NUM_CLASSES)
    class_center =torch.zeros(NUM_CLASSES, hidden_dim).to(device)
    c_criterion = c_criterion.to(device)

    # optimizer_center = optim.SGD(c_criterion.parameters(), lr=args.lam)

    logger = Logger(os.path.join(model_dir, 'log.txt'), title=args.arch)
    logger.set_names(['Learning Rate', 'Nat Val Loss', 'Nat Val Acc.'])
    # torch.load(class_center, 'centers.t')
    # if args.resume_model:
    #     resume_model=os.path.join(model_dir, 'ours-model-epoch{}.pt'.format(args.start_epoch-1))
    #     # resume_model1 = os.path.join(model_dir, 'ours-Gmodel-epoch{}.pt'.format(args.start_epoch-1))
    #     # resume_model2 = os.path.join(model_dir, 'ours-Gmodel2-epoch{}.pt'.format(args.start_epoch - 1))
    #     model.load_state_dict(torch.load(resume_model, map_location=device))
    #     # a_G.load_state_dict(torch.load(resume_model1, map_location=device))
    #     # a_G2.load_state_dict(torch.load(resume_model2, map_location=device))
    # if args.resume_optim:
    #     resume_optim = os.path.join(model_dir, 'ours-opt-checkpoint_epoch{}.tar'.format(args.start_epoch - 1))
    #     # resume_optim1 = os.path.join(model_dir, 'ours-optG-checkpoint_epoch{}.tar'.format(args.start_epoch - 1))
    #     # resume_optim2 = os.path.join(model_dir, 'ours-optG2-checkpoint_epoch{}.tar'.format(args.start_epoch - 1))
    #     optimizer.load_state_dict(torch.load(resume_optim, map_location=device))
    # optim_G.load_state_dict(torch.load(resume_optim1, map_location=device))
    # optim_G2.load_state_dict(torch.load(resume_optim2, map_location=device))
    start_wa = [(150 * args.epochs) // 200]
    tau_list = [0.9996]
    exp_avgs = []
    model_tau = getattr(models, args.arch)(num_classes=NUM_CLASSES).to(device)
    exp_avgs.append(model_tau.state_dict())
    num = int(args.mix_num)
    best_acc = 0.
    for epoch in range(args.start_epoch, args.epochs + 1):
        # lr = adjust_learning_rate_cosine(optimizer, epoch, args)
        lr = adjust_learning_rate(optimizer, epoch)
        # lr = adjust_learning_rate(optim_G, epoch)
        # lr = adjust_learning_rate(optim_G2, epoch)
        #
        adv_loss, adv_acc, exp_avgs, class_center = adv_train(model, c_criterion, class_center, train_loader, optimizer,
                                                              epoch,
                                                              awp_adversary, exp_avgs)

        print('================================================================')
        val_loss, val_acc = test(model, test_loader, criterion)
        print('================================================================')
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        logger.append([lr, val_loss, val_acc])

        if best_acc <= val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'best-model.pt'))

            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'best-opt-checkpoint.tar'))

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'ours-model-epoch{}.pt'.format(epoch)))

            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'ours-opt-checkpoint_epoch{}.tar'.format(epoch)))

            torch.save(class_center, os.path.join(model_dir, 'ours-centers.t'.format(epoch)))

        if epoch >= args.epochs - 1:
            for idx, start_ep, tau, new_state_dict in zip(range(len(tau_list)), start_wa, tau_list, exp_avgs):
                if start_ep <= epoch:
                    torch.save(new_state_dict,
                               os.path.join(model_dir, 'ours-model-epoch-SWA{}{}{}.pt'.format(tau, start_ep, epoch)))
        # scheduler.step()


if __name__ == '__main__':
    main()
