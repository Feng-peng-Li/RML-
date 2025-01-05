import os
import os.path
import argparse
import random
import numpy as np
import math

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LambdaLR
from torchvision.datasets import CIFAR10, CIFAR100
import torch.nn.functional as F
from networks.ResNet import ResNet18, ResNet34
from common.tools import getTime, evaluate, predict_softmax, train, train_warm, train_correction
from common.NoisyUtil import Train_Dataset, Semi_Labeled_Dataset, Semi_Unlabeled_Dataset, dataset_split
from RandAugment import RandAugment
import csv

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--u_bs', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.03, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=5e-4)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--lam', default=0.9, type=float, help='corruption rate, should be less than 1')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--data_path', type=str, default='./data', help='data directory')
parser.add_argument('--data_percent', default=1.0, type=float, help='data number percent')
parser.add_argument('--noise_type', default='pairflip', type=str)
parser.add_argument('--noise_rate', default=0.45, type=float, help='corruption rate, should be less than 1')
parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--warm_step', default=50, type=float, help='steps for warming')
parser.add_argument('--correction_step', default=200, type=float, help='steps for warming')
parser.add_argument('--n', default=14, type=float, help='weight for unsupervised loss')
parser.add_argument('--gpuid', default=4, type=float, help='weight for unsupervised loss')

args = parser.parse_args()
print(args)
os.system('nvidia-smi')

args.model_dir = 'model/'
if not os.path.exists(args.model_dir):
    os.system('mkdir -p %s' % (args.model_dir))

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # cudnn.deterministic = True
    cudnn.benchmark = True
device = torch.device('cuda:' + str(args.gpuid) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.gpuid)

def linear_rampup(current, warm_up=20, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


# MixMatch Training
def MixMatch_train(epoch, net,net2, optimizer,optimizer2, labeled_trainloader, unlabeled_trainloader, class_weights):
    net.train()
    if epoch >= args.num_epochs/2:
        args.alpha = 0.75

    losses = AverageMeter('Loss', ':6.2f')
    losses_lx = AverageMeter('Loss_Lx', ':6.2f')
    losses_lu = AverageMeter('Loss_Lu', ':6.5f')

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = int(50000/args.batch_size)
    for batch_idx in range(num_iter):
        try:
            inputs_x, inputs_x2, targets_x = labeled_train_iter.next()
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, inputs_x2, targets_x = labeled_train_iter.next()

        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()

        batch_size = inputs_x.size(0)
        targets_x = torch.zeros(batch_size, args.num_class).scatter_(1, targets_x.view(-1, 1), 1)
        inputs_x, inputs_x2, targets_x = inputs_x.cuda(), inputs_x2.cuda(), targets_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2
            ptu = pu**(1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixmatch_l = np.random.beta(args.alpha, args.alpha)
        mixmatch_l = max(mixmatch_l, 1 - mixmatch_l)

        mixed_input = mixmatch_l * input_a + (1 - mixmatch_l) * input_b
        mixed_target = mixmatch_l * target_a + (1 - mixmatch_l) * target_b

        logits = net(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        Lx_mean = -torch.mean(F.log_softmax(logits_x, dim=1) * mixed_target[:batch_size * 2], 0)
        Lx = torch.sum(Lx_mean * class_weights)

        probs_u = torch.softmax(logits_u, dim=1)
        Lu = torch.mean((probs_u - mixed_target[batch_size * 2:])**2)
        loss = Lx + linear_rampup(epoch + batch_idx / num_iter, args.T1) * Lu

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer2.step()
        losses_lx.update(Lx.item(), batch_size * 2)
        losses_lu.update(Lu.item(), len(logits) - batch_size * 2)
        losses.update(loss.item(), len(logits))

    print(losses, losses_lx, losses_lu)


def create_model(name="resnet18", input_channel=3, num_classes=10, ema=False):
    if (name == "resnet18"):
        model = ResNet18(num_classes)
    else:
        print("create ResNet34")
        model = ResNet34(num_classes)
    if ema:
        for param in model.parameters():
            param.detach_()
    model.cuda()
    return model


def splite_confident(outs, clean_targets, noisy_targets, tau_t, p_t, label_update, epcoh):
    probs, preds = torch.max(outs.data, 1)

    confident_correct_num = 0
    unconfident_correct_num = 0
    confident_indexs = []
    unconfident_indexs = []
    mom = 0.9
    clean_targets = torch.from_numpy(np.array(train_clean_labels)).cuda()
    c_label_update = label_update / torch.sum(label_update, dim=1, keepdim=True)
    t_l = torch.argmax(c_label_update.cpu(), dim=1)

    max_probs = probs
    b = torch.max(p_t, dim=-1)[0]
    tau_t_c = (p_t / b)
    mask_prob = max_probs.ge(tau_t * tau_t_c[preds])

    for i in range(0, len(noisy_targets)):
        if noisy_targets[i] == preds[i]:
            label_update[i] = F.one_hot(t_l[i], label_update.size(1)).float()
            if mask_prob[i] == True:
                confident_indexs.append(i)
                if clean_targets[i] == preds[i]:
                    confident_correct_num += 1
            else:
                unconfident_indexs.append(i)

                if clean_targets[i] == preds[i]:
                    unconfident_correct_num += 1

        else:
            # if epoch==150:
            #     if mask_prob[i] == True and t_l[i]==preds[i]:
            #         noisy_targets[i]=t_l[i]
            unconfident_indexs.append(i)
            if clean_targets[i] == preds[i]:
                unconfident_correct_num += 1
    label_update = label_update * mom + (1 - mom) * outs.cuda()
    r_l = torch.argmax(label_update, dim=1)
    u_acc = torch.eq(r_l, clean_targets).float().mean()
    print("clean split ratio:", confident_correct_num / len(confident_indexs))
    print("clean ratio:", u_acc)
    # print(getTime(), "confident and unconfident num:", len(confident_indexs), round(confident_correct_num / len(confident_indexs) * 100, 2), len(unconfident_indexs), round(unconfident_correct_num / len(unconfident_indexs) * 100, 2))
    return confident_indexs, unconfident_indexs, label_update


class LRScheduler:

    def __init__(
            self,
            optimizer,
            num_train_iters=1,
            num_warm_up=10,
            num_cycles=7. / 16
    ):
        self.num_train_iters = num_train_iters
        self.num_cycles = num_cycles
        self.num_warmup_iters = num_warm_up

        self.scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=self.__lr__step__,
            last_epoch=-1
        )

    def step(self):

        self.scheduler.step()

    def __lr__step__(self, current_step):

        if current_step < self.num_warmup_iters:
            _lr = float(current_step) / float(max(1, self.num_warmup_iters))
        else:
            num_cos_steps = float(current_step - self.num_warmup_iters)
            num_cos_steps = num_cos_steps / float(max(1, self.num_train_iters - self.num_warmup_iters))
            _lr = max(0.0, math.cos(math.pi * self.num_cycles * num_cos_steps))
        return _lr


def update_trainloader(model, model1, train_data, clean_targets, noisy_targets, tau_t, p_t, label_update, epoch):
    predict_dataset = Semi_Unlabeled_Dataset(train_data, noisy_targets, label_update.cpu().numpy(),
                                             [transform_train, transform_train_rand])
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=2,
                                pin_memory=True, drop_last=False)
    soft_outs = predict_softmax(predict_loader, model1)

    confident_indexs, unconfident_indexs, label_update = splite_confident(soft_outs, clean_targets, noisy_targets,
                                                                          tau_t, p_t, label_update, epoch)
    confident_dataset = Semi_Labeled_Dataset(train_data[confident_indexs], noisy_targets[confident_indexs],
                                             transform_train)
    unconfident_dataset = Semi_Unlabeled_Dataset(train_data[unconfident_indexs], noisy_targets[unconfident_indexs],
                                                 label_update[unconfident_indexs].cpu().numpy(),
                                                 [transform_train, transform_train_rand])

    # uncon_batch = int(args.batch_size / 2) if len(unconfident_indexs) > len(confident_indexs) else int(
    #     len(unconfident_indexs) / (len(confident_indexs) + len(unconfident_indexs)) * args.batch_size)
    # con_batch = args.batch_size - uncon_batch

    # labeled_trainloader = DataLoader(dataset=confident_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
    #                                  pin_memory=True, drop_last=True)
    # unlabeled_trainloader = DataLoader(dataset=unconfident_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
    #                                    pin_memory=True, drop_last=True)

    # Loss function
    train_nums = np.zeros(args.num_class, dtype=int)
    for item in noisy_targets[confident_indexs]:
        train_nums[item] += 1

    # zeros are not calculated by mean
    # avoid too large numbers that may result in out of range of loss.
    with np.errstate(divide='ignore'):
        cw = np.mean(train_nums[train_nums != 0]) / train_nums
        cw[cw == np.inf] = 0
        cw[cw > 3] = 3
    class_weights = torch.FloatTensor(cw).cuda()
    # print("Category", train_nums, "precent", class_weights)
    return confident_dataset, unconfident_dataset, class_weights, label_update


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        # self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            # fix the error 'RuntimeError: result type Float can't be cast to the desired output type Long'
            # print(param.type())
            if param.type() == 'torch.cuda.LongTensor':
                ema_param = param
            else:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)


if args.dataset == 'cifar10' or args.dataset == 'CIFAR10':

    args.num_class = 10
    args.model_name = "resnet18"
    transform_train = transforms.Compose([

        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train_rand = transforms.Compose([
        RandAugment(3),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = CIFAR10(root=args.data_path, train=True, download=True)
    test_set = CIFAR10(root=args.data_path, train=False, transform=transform_test, download=True)
elif args.dataset == 'cifar100' or args.dataset == 'CIFAR100':

    args.num_class = 100
    args.model_name = "resnet34"
    transform_train = transforms.Compose([
        RandAugment(3),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    transform_train_rand = transforms.Compose([
        RandAugment(3),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
    train_set = CIFAR100(root=args.data_path, train=True, download=True)
    test_set = CIFAR100(root=args.data_path, train=False, transform=transform_test, download=True)

train_data, val_data, train_noisy_labels, val_noisy_labels, train_clean_labels, _ = dataset_split(train_set.data,
                                                                                                  np.array(
                                                                                                      train_set.targets),
                                                                                                  args.noise_rate,
                                                                                                  args.noise_type,
                                                                                                  args.data_percent,
                                                                                                  args.seed,
                                                                                                  args.num_class, False)
train_dataset = Train_Dataset(train_data, train_noisy_labels, transform_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                          pin_memory=True, drop_last=True)
val_dataset = Train_Dataset(val_data, val_noisy_labels, transform_train)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=2,
                        pin_memory=True)
test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=2,
                         pin_memory=True)

model = create_model(name=args.model_name, num_classes=args.num_class)
model1 = create_model(name=args.model_name, num_classes=args.num_class, ema=True)
model2 = create_model(name=args.model_name, num_classes=args.num_class, ema=True)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, args.num_epochs, args.lr / 100)

# scheduler = LRScheduler(optimizer=optimizer,num_train_iters=args.num_epochs,num_warm_up=args.warm_step)

ceriation = nn.CrossEntropyLoss(reduction='none').cuda()
train_ceriation = ceriation
optimizer2 = WeightEMA(model, model1)
best_val_acc = 0
best_test_acc = 0
class_center = F.normalize(F.softmax(torch.rand(args.num_class, args.num_class, dtype=torch.float), dim=1), dim=1,
                           p=2).cuda(non_blocking=True)
label_update = F.one_hot(torch.from_numpy(np.array(train_clean_labels)).cuda(), args.num_class).float()
loss_m, label_m, label_p = torch.zeros(len(train_set), args.num_class).cuda(non_blocking=True), torch.zeros(
    len(train_set),
    args.num_class).cuda(non_blocking=True), torch.zeros(len(train_set), args.num_class).cuda(non_blocking=True)
confident_dataset, unconfident_dataset = 0., 0.
c_ind, u_ind = 0., 0.
tau_t = torch.ones(1).cuda() * (1. / args.num_class)
p_t = (torch.ones(args.num_class).cuda()) * tau_t
label_hist = (torch.ones(args.num_class).cuda()) * tau_t

train_acc=[]
test_acc=[]
for epoch in range(args.num_epochs):
    if epoch < args.warm_step:
        _, _, label_update = train_warm(model, model1, train_loader, optimizer, optimizer2, ceriation, epoch, args.n,
                                        args.num_class, label_update)
    else:
        if epoch == args.warm_step or epoch % 1 == 0:
            clean_targets = torch.from_numpy(np.array(train_clean_labels)).cuda()

            confident_dataset_n, unconfident_dataset_n, class_weights, label_update = update_trainloader(model1,
                                                                                                         model1,
                                                                                                         train_data,
                                                                                                         train_clean_labels,
                                                                                                         train_noisy_labels,
                                                                                                         tau_t.cpu(),
                                                                                                         p_t.cpu(),
                                                                                                         label_update,
                                                                                                         epoch)

            labeled_trainloader = DataLoader(dataset=confident_dataset_n, batch_size=args.batch_size, shuffle=True,
                                             num_workers=2, pin_memory=True, drop_last=True)

            unlabeled_trainloader = DataLoader(dataset=unconfident_dataset_n, batch_size=args.u_bs, shuffle=True,
                                               num_workers=2, pin_memory=True, drop_last=True)
            if epoch < args.correction_step:

                # _, _, loss_m, label_m, label_p, tau_t, p_t, label_hist, label_update = train(model, model1, train_loader,
                #                                                                              clean_targets, optimizer,
                #                                                                              optimizer2, ceriation, epoch,
                #                                                                              args.n, args.num_class, tau_t,
                #                                                                              p_t, label_hist, ema_w=0.999,
                #                                                                              label_update=label_update)
                _, _, tau_t, p_t, label_hist = train(model, model1, labeled_trainloader, unlabeled_trainloader,
                                                     optimizer, optimizer2, ceriation,
                                                     epoch,
                                                     args.n, args.num_class, class_weights, tau_t, p_t, label_hist,
                                                     ema_w=0.999)
            else:

                MixMatch_train(epoch, model, model1, optimizer, optimizer2, labeled_trainloader, unlabeled_trainloader,
                               class_weights)
            _, val_acc = evaluate(model, val_loader, ceriation, "Val Acc:")

    scheduler.step()
    _,t_acc=evaluate(model, test_loader, ceriation, "train Acc:")
    _, val_acc = evaluate(model, test_loader, ceriation, "Val Acc:")
    _, val_acc1 = evaluate(model1, test_loader, ceriation, "Val Acc:")
    train_acc.append(t_acc)
    test_acc.append(val_acc)

    # if val_acc>=best_val_acc:
    #     _, test_acc = evaluate(model, test_loader, ceriation, "Epoch " + str(epoch) + " Test Acc:")
    #     best_test_acc = test_acc
with open(args.noise_type+'_'+args.dataset+'_'+str(args.noise_rate)+'_'+'tra_acc.csv','w',newline='',encoding='utf-8') as csvfile1:
    writer = csv.writer(csvfile1)
    writer.writerow(train_acc)
with open(args.noise_type+'_'+args.dataset+'_'+str(args.noise_rate)+'_'+'tes_acc.csv','w',newline='',encoding='utf-8') as csvfile2:
    writer = csv.writer(csvfile2)
    writer.writerow(test_acc)
print(getTime(), "Best Test Acc:", best_test_acc)
