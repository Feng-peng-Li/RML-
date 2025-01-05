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
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
from PIL import Image

from models.resnet import ResNet18
import models

import torchvision
import kornia.augmentation as K

import random
from Tiny import TinyImageNet
from torchvision.utils import save_image
from kornia.augmentation import AugmentationSequential

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--arch', type=str, default='ResNet18t')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--js_weight', default=2.0, type=int, metavar='N',
                    help='The weight of the JS divergence term')
parser.add_argument('--mask_t', default=0.4, type=int,
                    help='mask_t')
parser.add_argument('--loss', default='mse', type=str,
                    help='perturbation', choices=['mse', 'M', 'cosine'])
parser.add_argument('--c_epoch', type=int, default=50, metavar='N',
                    help='retrain from which epoch')
parser.add_argument('--gpu_id', default=1, type=int,
                    help='gpu')
parser.add_argument('--ksize', type=int, default=2, metavar='ksize',
                    help='ksize')
parser.add_argument('--p_size', type=int, default=4, metavar='psize',
                    help='psize')
parser.add_argument('--beta', default=12, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--mix_beta', default=0.01, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--w', default=6., type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--lam', type=int, default=0.9, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--lam_alpha', type=float, default=0.7
                    , metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--step1', default=70, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--step2', default=75, type=float,
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
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of epochs to train')

parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
parser.add_argument('--data-path', type=str, default='./data/tiny-imagenet-200/',
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
                    default='/home/yc17420/DAJAT-main/model-cifar-ResNet_0_50/ours-Gmodel-epoch119.pt', type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim1',
                    default='/home/yc17420/DAJAT-main/model-cifar-ResNet_0_50/ours-optG-checkpoint_epoch119.tar',
                    type=str,
                    help='directory of optimizer for retraining')
parser.add_argument('--resume-model2',
                    default='/home/yc17420/DAJAT-main/model-cifar-ResNet_0_50/ours-Gmodel2-epoch119.pt', type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim2',
                    default='/home/yc17420/DAJAT-main/model-cifar-ResNet_0_50/ours-optG2-checkpoint_epoch199.tar',
                    type=str,
                    help='directory of optimizer for retraining')
parser.add_argument('--save-freq', '-s', default=100, type=int, metavar='N',
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

NUM_CLASSES = 200

# settings
model_dir = args.model_dir + '_' + str(args.gpu_id) + '_' + str(args.mix_num)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.gpu_id)

kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}


# dist.init_process_group(backend='nccl')
# local_rank = int(os.environ['LOCAL_RANK'])
# torch.cuda.set_device(local_rank)
#
# # device = torch.device("cuda", local_rank)
# verbose = dist.get_rank() == 0

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CropShift(torch.nn.Module):
    def __init__(self, low, high=None):
        super().__init__()
        high = low if high is None else high
        self.low, self.high = int(low), int(high)

    def sample_top(self, x, y):
        x = torch.randint(0, x + 1, (1,)).item()
        y = torch.randint(0, y + 1, (1,)).item()
        return x, y

    def forward(self, img):
        if self.low == self.high:
            strength = self.low
        else:
            strength = torch.randint(self.low, self.high, (1,)).item()

        w, h = TF.get_image_size(img)
        crop_x = torch.randint(0, strength + 1, (1,)).item()
        crop_y = strength - crop_x
        crop_w, crop_h = w - crop_x, h - crop_y

        top_x, top_y = self.sample_top(crop_x, crop_y)

        img = TF.crop(img, top_y, top_x, crop_h, crop_w)
        img = TF.pad(img, padding=[crop_x, crop_y], fill=0)

        top_x, top_y = self.sample_top(crop_x, crop_y)

        return TF.crop(img, top_y, top_x, h, w)


transform_train = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.RandomCrop(64, padding=2),
    transforms.RandomHorizontalFlip(),
    # CropShift(0, 22),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
])


# IDBH(idbh_p,args.size)

class TinyDataset(torchvision.datasets.ImageFolder):

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)
        image = self.transform(img)
        return image, target, idx


# trainset = TinyDataset(os.path.join(args.data_path, 'train'), transform_train)
# testset = TinyDataset(os.path.join(args.data_path, 'val'), transform_test)


trainset = TinyImageNet(args.data_path, split='train', transform=transform_train, in_memory=True, )
# # trainset=TinyDataset(root=os.path.join(args.data_path, 'train'), transform=[IDBH(idbh_p), transform_train])
#
testset = TinyImageNet(args.data_path, split='val', transform=transform_test, in_memory=True)
# testset = datasets.ImageFolder(root=os.path.join(args.data_path, 'val/val_img'),  transform=transform_train)
# valset = torchvision.datasets.ImageFolder(root=args.data_path, split='val', transform=[transform_test])

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
# # val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,sampler=SubsetRandomSampler(val_indices), **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


# train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
# test_loader =torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

def symmkl(upper_logit, lower_logit):
    loss_1 = F.kl_div(F.log_softmax(upper_logit, dim=1),
                      F.softmax(lower_logit.data, dim=1),
                      reduction="batchmean")
    loss_2 = F.kl_div(F.log_softmax(lower_logit, dim=1),
                      F.softmax(upper_logit.data, dim=1),
                      reduction="batchmean")
    return 0.5 * (loss_1 + loss_2)


def update_center(train_loader, model, class_center, lam):
    model.eval()
    feature_all = []
    target_all = []
    for batch_idx, (data, _, target, index) in enumerate(train_loader):
        x_natural, target = data.to(device), target.to(device)
        logits, feature, _ = model(x_natural, if_f=True)
        feature_all.append(torch.softmax(feature.detach(), dim=1))
        target_all.append(target.detach())

    feature_all = torch.cat(feature_all, dim=0)
    target_all = torch.cat(target_all, dim=0)

    for i in range(NUM_CLASSES):
        index_i = torch.nonzero(target_all == i)
        feature_i = feature_all[index_i]
        feature_i = torch.mean(feature_i, dim=0, keepdim=True)
        class_center[i:i + 1] = (1 - lam) * feature_i + lam * class_center[i:i + 1]
    class_center = F.normalize(class_center, dim=1, p=2)
    # class_center = torch.log_softmax(class_center, dim=1)
    return class_center


class Custom_CrossEntropy_PSKD(nn.Module):
    def __init__(self):
        super(Custom_CrossEntropy_PSKD, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, output, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(output)
        loss = (- targets * log_probs).mean(0).sum()
        return loss


def obtain_c_soft(center, feature):
    batch_size = feature.size(0)
    center = center.unsqueeze(0).repeat(batch_size, 1, 1)
    c_feature = feature.unsqueeze(1).repeat(1, NUM_CLASSES, 1)
    distance = torch.sum((c_feature - center) ** 2, dim=-1)
    soft_target = F.softmax(-distance / args.mix_beta, dim=1)
    return soft_target


def perturb_input(model,
                  x_natural,
                  class_center,
                  target,
                  target_soft,
                  target_alpha,
                  c_center,
                  epoch,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf',
                  no_core=True):
    model.eval()
    batch_size = len(x_natural)
    CE = Custom_CrossEntropy_PSKD()
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()

        for _ in range(perturb_steps):
            x_adv.requires_grad_()

            with torch.enable_grad():
                logits_adv, feature_f, _ = model(x_adv, if_f=True)
                logits_na = model(x_natural)
                if no_core:
                    if epoch >= 0:
                        loss_contrast = args.w * c_center(
                            F.softmax(feature_f, dim=1), class_center, target)

                        # loss_kl1=CE(logits_adv,target_soft)+loss_contrast
                        loss_kl1 = args.beta * symmkl(logits_adv, logits_na) + loss_contrast
                    else:
                        loss_kl1 = symmkl(logits_adv, logits_na)
                else:
                    loss_kl1 = symmkl(logits_adv, logits_na)
            grad_na = torch.autograd.grad(loss_kl1, [x_adv])[0]

            x_adv = x_adv.detach() + step_size * torch.sign(grad_na.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv


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
        one_y = F.one_hot(y, centers.size(0))

        expanded_centers = centers.expand(batch_size, -1, -1)
        expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1, 0)
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1)
        epsilon = 1e-6
        dis_same = torch.sum(one_y * distance_centers, dim=1)
        dis_diff = torch.sum(torch.exp(distance_centers), dim=1)
        loss = torch.mean(-torch.log(1 - torch.exp(dis_same) / dis_diff), dim=0)
        return loss


def adv_train(model, c_criterion, class_center, soft_label, label_rem, train_loader, optimizer, epoch, awp_adversary):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    CE = Custom_CrossEntropy_PSKD()
    init_step_size = 2. / 255
    init_epsilon = 8. / 255
    # init_epsilon * (epoch / args.epochs)
    epsilon = init_epsilon
    step_size = init_step_size
    num_steps = 10
    print('epoch: {}'.format(epoch))
    bar = Bar('Processing', max=len(train_loader))

    alpha_t = ((epoch + 1) / args.epochs)
    alpha_t = min(args.lam, alpha_t)
    # alpha_t = args.lam_alpha * alpha_t
    K_aug = AugmentationSequential(
        K.RandomHorizontalFlip(),
        K.RandomCrop((32, 32), cropping_mode='resample', padding=args.ksize),
    )
    target_all = []
    feature_all = []
    feature_core_all = []

    for batch_idx, (data_base, target, index) in enumerate(train_loader):

        x_natural, target, index = data_base.to(device), target.to(device), index.to(device)
        # print(index)
        target_all.append(target)
        target_one = F.one_hot(target, NUM_CLASSES)
        if epoch >= args.c_epoch:
            # model.train()
            batch_feature = soft_label[index]
            soft_pred = obtain_c_soft(center=class_center, feature=batch_feature)
            soft_target = target_one * (1 - alpha_t) + alpha_t * soft_pred
            # soft_target = target_one

            # soft_pred1 = obtain_c_soft(center=class_center, feature=soft_label1[index])
            # # # # # # soft_pred=soft_tar**2+target_one.detach()-torch.mean(soft_tar**2,dim=0,keepdim=True)
            # soft_target1 = target_one * (1 - alpha_t) + alpha_t * soft_pred1


        else:
            soft_target = target_one
            soft_target1 = target_one

        x_adv = perturb_input(model=model,
                              x_natural=x_natural,
                              class_center=class_center,
                              target=target,
                              target_soft=soft_target,
                              target_alpha=alpha_t,
                              epoch=epoch,
                              c_center=c_criterion,
                              step_size=step_size,
                              epsilon=epsilon,
                              perturb_steps=num_steps,
                              distance=args.norm,
                              no_core=True)
        model.train()

        logits_adv_na, adv_na_feature, f_map_adv = model(x_adv, if_f=True)
        logits_na, feature_na, f_map_na = model(x_natural, if_f=True)

        # loss_robust_na = F.kl_div(F.log_softmax(logits_adv_na, dim=1),
        #                           F.softmax(logits_na, dim=1),
        #                           reduction='batchmean')
        loss_robust_na = symmkl(logits_adv_na, logits_na)
        loss_natural = CE(logits_na, soft_target)

        if epoch >= args.awp_warmup:
            awp = awp_adversary.calc_awp(inputs_adv=x_adv, inputs_clean=x_natural, targets=target, beta=args.beta,
                                         smoothing=args.lam_alpha)
            awp_adversary.perturb(awp)
        optimizer.zero_grad()
        if epoch >= 0:
            loss_contrast = args.w * (c_criterion(
                F.softmax(adv_na_feature, dim=1), class_center, target))
            loss = (loss_natural + args.beta * loss_robust_na) + loss_contrast
        else:
            loss = loss_natural + args.beta * loss_robust_na

        # + args.beta * loss_robust_core + args.beta * loss_robust_na
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)

        prec1, prec5 = accuracy(logits_adv_na, target, topk=(1, 5))
        losses.update(loss.item(), x_natural.size(0))
        top1.update(prec1.item(), x_natural.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        for i in range(NUM_CLASSES):
            if i in target:
                index_i = torch.nonzero(target == i)
                feature_i = F.softmax(feature_na.detach(), dim=1)[index_i]
                feature_i = torch.mean(feature_i, dim=0, keepdim=True)
                class_center[i:i + 1] = (1 - args.lam) * feature_i + args.lam * class_center[i:i + 1]
                class_center[i:i + 1] = F.normalize(class_center[i:i + 1], dim=1, p=2)

        soft_label[index] = args.lam * soft_label[index] + (1 - args.lam) * F.softmax(feature_na.detach(),
                                                                                      dim=1).detach()

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total:{total:}| ETA:{eta:}| Loss:{loss:.4f}| top1:{top1:.2f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg)
        bar.next()

    # if epoch >= 0:
    #     class_center=update_center(train_loader, model, class_center,args.lam)

    bar.finish()
    return losses.avg, top1.avg, class_center


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

    for batch_idx, (inputs, targets, _) in enumerate(test_loader):
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
    if epoch >= args.step1:
        lr = args.lr * 0.1
    if epoch >= args.step2:
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
    class_center = F.normalize(torch.randn(NUM_CLASSES, hidden_dim), dim=1, p=2).to(device)
    soft_label = torch.zeros(len(trainset), args.in_ch, dtype=torch.float).cuda(non_blocking=True)
    label_rem = torch.zeros(len(trainset), NUM_CLASSES, dtype=torch.float).cuda(non_blocking=True)
    c_criterion = c_criterion.to(device)

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
        adv_loss, adv_acc, class_center = adv_train(model, c_criterion, class_center, soft_label, label_rem,
                                                    train_loader, optimizer,
                                                    epoch,
                                                    awp_adversary)

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

        if epoch % args.save_freq == 0 or epoch == args.step2:
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

