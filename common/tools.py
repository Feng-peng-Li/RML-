import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn


def getTime():
    time_stamp = datetime.datetime.now()
    return time_stamp.strftime('%H:%M:%S')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def regroup_median_split(loss_data, label_u, n, loss_m, loss_u, label_m, s_index_u):
    loss_v = loss_data
    loss = []
    b_s = loss_data.size(0)
    for i in range(b_s):

        index = torch.nonzero(label_m == label_u[i]).squeeze()
        if len(index.size()) != 0:
            loss_x = loss_m[index]
            # r_idx = torch.randint(0, loss_x.size(0), [loss_x.size(0)]).to(label_u.device)
            # loss_x = loss_x[r_idx]
            r_idx = torch.randperm(loss_x.size(0))
            loss_x = loss_x[r_idx]
            if len(loss_x.size()) != 0:
                e_num = (int(loss_x.size(0) / n) + 1) * n - loss_x.size(0)
                if e_num <= (n / 2):
                    if e_num > 0:
                        loss_norm = torch.exp(-loss_x ** 2 - loss_x)
                        loss_norm = loss_norm / (loss_norm.sum() + 0.00000001)
                        select_index = torch.multinomial(loss_norm, e_num, replacement=False)
                        select_loss = loss_x[select_index].detach()
                        loss_x = torch.cat([loss_x, select_loss], dim=0)

                        loss_x_i_1 = loss_u[s_index_u[i]].detach()

                    loss_x = loss_x.view(n, int(loss_x.size(0) / n))
                    select_loss_mean = torch.mean(loss_x, dim=1)
                    select_loss = torch.cat([loss_x_i_1.view(1), select_loss_mean])

                    select_loss = torch.median(select_loss)
                    l_weight = select_loss / (loss_x_i_1 + 0.0000001)
                    if loss_v[i].detach() <= loss_v[i].detach() * l_weight or loss_v[i] == 0:
                        loss.append(loss_v[i].view(1))

                    else:
                        tmp = l_weight * loss_v[i]
                        loss.append(tmp.view(1))

                else:
                    loss_data[i] = 0.
                    loss.append(loss_data[i].view(1))
            else:
                loss_data[i] = 0.
                loss.append(loss_data[i].view(1))
        else:
            # loss_data[i] = 0.
            loss.append(loss_data[i].view(1))
    if len(loss) != 0:
        loss = torch.cat(loss, dim=0)
    else:
        loss = loss_data.sum() / s_index_u.size(0)
    return loss


def filter_split(model, train_loader, ceriation, num_class, flag=False):
    label_m = torch.zeros(50000).cuda()
    label_p = torch.zeros(50000).cuda()
    loss_m = torch.zeros(50000).cuda()
    loss_m_u = torch.zeros(50000, num_class).cuda()
    logits = torch.zeros(50000, num_class).cuda()
    with torch.no_grad():
        for i, (images, _, labels, index) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            if labels.dim() > 1:

                loss_m[index] = -torch.sum(torch.log_softmax(output, dim=1) * labels, dim=1).float()
            else:
                if flag:
                    loss_m_u[index] = -F.log_softmax(output, dim=1)
                loss_m[index] = ceriation(output, labels.long())
            label_m[index] = labels.float()
            pred_label = torch.argmax(torch.softmax(output, dim=1), dim=1)
            label_p[index] = pred_label.float()
            logits[index] = F.softmax(output, dim=1)

    return loss_m, label_m, label_p, loss_m_u


def filter_noisy(model, train_loader, ceriation, num_class, flag=False):
    label_m = torch.zeros(50000).cuda()
    label_p = torch.zeros(50000).cuda()
    loss_m = torch.zeros(50000).cuda()
    loss_m_u = torch.zeros(50000, num_class).cuda()
    logits = torch.zeros(50000, num_class).cuda()
    with torch.no_grad():
        for i, (images, _, labels, _, index) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            if labels.dim() > 1:

                loss_m[index] = -torch.sum(torch.log_softmax(output, dim=1) * labels, dim=1).float()
            else:
                if flag:
                    loss_m_u[index] = -F.log_softmax(output, dim=1)
                loss_m[index] = ceriation(output, labels.long())
            label_m[index] = labels.float()
            pred_label = torch.argmax(torch.softmax(output, dim=1), dim=1)
            label_p[index] = pred_label.float()
            logits[index] = F.softmax(output, dim=1)

    return loss_m, label_m, label_p, loss_m_u


class ConsistencyLoss:

    def __call__(self, logits, targets, mask=None):
        preds = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(preds, targets, reduction='none')
        if mask is not None:
            masked_loss = loss * mask.float()
            return masked_loss.mean()
        return loss.mean()


class Custom_CrossEntropy_PSKD(nn.Module):
    def __init__(self):
        super(Custom_CrossEntropy_PSKD, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, output, targets, reduction='mean'):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(output)
        if reduction == 'mean':
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(dim=1)
        return loss


def train_correction(model, model1, labeled_trainloader, unlabeled_trainloader, optimizer, optimizer2, ceriation, epoch,
                     step, num_classes, class_weights, tau_t, p_t, label_hist, ema_w):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    losses = AverageMeter('Loss', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(labeled_trainloader),
        [batch_time, data_time, losses, top1],
        prefix="Train Epoch: [{}]".format(epoch))
    consis_criterion = ConsistencyLoss()
    model.train()
    CE = Custom_CrossEntropy_PSKD()
    end = time.time()
    loss_m, label_m, _, _ = filter_split(model, labeled_trainloader, ceriation, num_classes)
    loss_m_u, label_m_u, _, loss_m_u_all = filter_noisy(model, unlabeled_trainloader, ceriation, num_classes, flag=True)

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = len(labeled_trainloader)
    for batch_idx in range(num_iter):
        try:
            inputs_x, _, targets_x, index = next(labeled_train_iter)
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, _, targets_x, index = next(labeled_train_iter)

        try:
            inputs_u, inputs_u2, target_u, _, index_u = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, target_u, _, index_u = next(unlabeled_train_iter)

        inputs_x, inputs_u, inputs_u2, targets_x, target_u = inputs_x.cuda(), inputs_u.cuda(), inputs_u2.cuda(), targets_x.cuda(), target_u.cuda()
        batch_size = inputs_x.size(0)
        u_bs = inputs_u.size(0)

        inputs_all = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)

        logit_all = model(inputs_all)

        loss_label = F.cross_entropy(logit_all[0:batch_size], targets_x)
        # loss_m[index]=loss_label.detach()

        loss_l = loss_label
        logit_u1 = logit_all[batch_size:batch_size + u_bs]

        prob_u = F.softmax(model1(inputs_u).detach(), dim=1)

        tau_t, p_t, label_hist = update_para(tau_t, p_t, prob_u, label_hist, ema_w)

        max_prob, max_target = torch.max(prob_u, dim=1)
        b = torch.max(p_t, dim=-1)[0]
        tau_t_c = (p_t / b)
        mask = max_prob.ge(tau_t * tau_t_c[max_target])

        update_index = torch.nonzero(mask == True).squeeze()
        target_u[update_index] = max_target[update_index]
        # target_u = max_target
        # target_u=torch.argmax(u_target,dim=1)
        logit_u2 = logit_all[batch_size + u_bs:batch_size + 2 * u_bs]
        loss_tmp = loss_m_u_all[index_u] * F.one_hot(target_u, num_classes).float()
        loss_m_u[index_u] = torch.sum(loss_tmp, dim=1)
        # u_target=u_target/torch.sum(u_target,dim=1,keepdim=True)

        loss_u2 = F.cross_entropy(logit_u2, target_u, reduction='none')
        # loss_u= CE(logit_u2,u_target,reduction='none')
        # loss_consis1 = regroup_median_split(loss_u, target_u, step, loss_m, loss_m_u, label_m, index_u)
        loss_r = regroup_median_split(loss_u2, target_u, step, loss_m, loss_m_u, label_m, index_u).mean()

        loss = (loss_r + loss_l)

        # loss_consis=consis_criterion(logit_u2,max_target,mask=mask)

        # loss=loss_l+loss_consis
        # if len(mask.size())!=0:
        #
        # loss_tmp=loss_m_u_all[index_u] * F.one_hot(target_u, num_classes).float()
        #
        # loss_m_u[index_u] = torch.sum(loss_tmp, dim=1)
        #
        #     logit_u2=logit_all[batch_size+u_bs:batch_size+2*u_bs]
        #     loss_u=F.cross_entropy(logit_u2[update_index],max_target[update_index],reduction='none')
        #
        #     loss_r_u=regroup_median_split(loss_u, max_target[update_index], step, loss_m, loss_m_u, label_m, index_u).mean()
        #     loss_hist,_=loss_fair(logit_u2[update_index],p_t,label_hist)
        #     loss=loss_l+loss_r_u
        # else:
        #     loss=loss_l

        acc1, acc5 = accuracy(logit_all[0:batch_size], targets_x, topk=(1, 5))
        losses.update(loss.item(), inputs_x[0].size(0))
        top1.update(acc1[0], inputs_x[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer2.step()
        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(0)
    return losses.avg, top1.avg.to("cpu", torch.float).item(), tau_t, p_t, label_hist


def regroup_median_matrix(loss_data, label, n, b_s, loss_m, label_m, s_index, c_ind, u_ind):
    loss_v = loss_data
    loss = []
    label_c = label_m[c_ind]
    loss_c = loss_m[c_ind]
    for i in range(b_s):
        if s_index[i] in u_ind:
            index = torch.nonzero(label_c == label[i]).squeeze()
            loss_x = loss_c[index]
            if len(loss_x.size()) != 0:

                # r_idx = torch.randint(0, loss_x.size(0), [loss_x.size(0)]).to(label.device)
                # loss_x = loss_x[r_idx]

                r_idx = torch.randperm(loss_x.size(0))
                loss_x = loss_x[r_idx]
                e_num = (int(loss_x.size(0) / n) + 1) * n - loss_x.size(0)

                if e_num <= (n / 2):
                    if e_num > 0:
                        loss_norm = torch.exp(-loss_x ** 2 - loss_x)
                        loss_norm = loss_norm / (loss_norm.sum() + 0.00000001)
                        select_index = torch.multinomial(loss_norm, e_num, replacement=False)
                        select_loss = loss_x[select_index].detach()
                        loss_x = torch.cat([loss_x, select_loss], dim=0)

                    loss_x_i_1 = loss_m[s_index[i]]
                    loss_x = loss_x.view(n, int(loss_x.size(0) / n))

                    select_loss_mean = torch.mean(loss_x, dim=-1)
                    select_loss = torch.cat([loss_x_i_1.view(1), select_loss_mean])

                    select_loss = torch.median(select_loss)
                    l_weight = select_loss / (loss_x_i_1 + 0.0000001)

                    if loss_v[i].detach() <= loss_v[i].detach() * l_weight or loss_v[i] == 0:
                        loss.append(loss_v[i].view(1))
                    else:
                        tmp = l_weight * loss_v[i]
                        loss.append(tmp.view(1))
                else:
                    loss_data[i] = 0.
                    loss.append(loss_data[i].view(1))
            else:
                loss_data[i] = 0.
                loss.append(loss_data[i].view(1))
        else:
            loss.append(loss_data[i].view(1))

    loss = torch.cat(loss, dim=0)
    return loss


def regroup_median(loss_data, label, n, b_s, loss_m, label_m, s_index):
    loss_v = loss_data
    loss = []
    for i in range(b_s):
        index = torch.nonzero(label_m == label[i]).squeeze()
        loss_x = loss_m[index]

        e_num = (int(loss_x.size(0) / n) + 1) * n - loss_x.size(0)
        # r_idx = torch.randint(0, loss_x.size(0), [loss_x.size(0)]).to(label.device)
        # loss_x = loss_x[r_idx]

        r_idx = torch.randperm(loss_x.size(0))
        loss_x = loss_x[r_idx]
        if e_num > 0:
            loss_norm = torch.exp(-loss_x ** 2 - loss_x)
            loss_norm = loss_norm / (loss_norm.sum() + 0.00000001)
            select_index = torch.multinomial(loss_norm, e_num, replacement=False)
            select_loss = loss_x[select_index].detach()
            loss_x = torch.cat([loss_x, select_loss], dim=0)

        loss_x_i_1 = loss_m[s_index[i]]
        loss_x = loss_x.view(n, int(loss_x.size(0) / n))
        select_loss_mean = torch.mean(loss_x, dim=1)

        select_loss = torch.cat([loss_x_i_1.view(1), select_loss_mean])
        select_loss = torch.median(select_loss)
        l_weight = select_loss / (loss_x_i_1 + 0.0000001)
        # print("estimate loss:",loss_v[i]* l_weight)
        # print("observed loss:",loss_v[i])
        if loss_v[i].detach() <= loss_v[i].detach() * l_weight or loss_v[i] == 0:
            loss.append(loss_v[i].view(1))
            # print(loss[i])
            # print(loss)
        else:
            tmp = l_weight * loss_v[i]
            loss.append(tmp.view(1))

    loss = torch.cat(loss, dim=0)
    return loss


def filter(model, model1, train_loader, ceriation, num_class, flag=False):
    label_m = torch.zeros(50000).cuda()
    label_p = torch.zeros(50000).cuda()
    loss_m = torch.zeros(50000).cuda()
    logits = torch.zeros(50000, num_class).cuda()
    logits1 = torch.zeros(50000, num_class).cuda()
    with torch.no_grad():
        for i, (images, labels, index) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            if labels.dim() > 1:
                loss_m[index] = -torch.mean(torch.sum(torch.log_softmax(output, dim=1) * labels, dim=1), dim=0).float()
            else:
                loss_m[index] = ceriation(output, labels.long())
            label_m[index] = labels.float()
            if flag:
                output1 = model1(images)
                pred_label = torch.argmax(torch.softmax(output1, dim=1), dim=1)
                logits[index] = F.softmax(output1, dim=1)

            else:
                pred_label = torch.argmax(torch.softmax(output, dim=1), dim=1)
                logits[index] = F.softmax(output, dim=1)
            label_p[index] = pred_label.float()

    return loss_m, label_m, label_p, logits


def loss_fair(logits_ulb_s, p_t, label_hist):
    probs_ulb_s = torch.softmax(logits_ulb_s, dim=-1)
    max_idx_s = torch.argmax(probs_ulb_s, dim=-1)

    # Calculate the histogram of strong logits acc. to Eq. 9
    # Cast it to the dtype of the strong logits to remove the error of division of float by long
    histogram = torch.bincount(max_idx_s, minlength=logits_ulb_s.shape[1]).to(logits_ulb_s.dtype)
    histogram /= histogram.sum()

    # Eq. 11 of the paper.
    p_t = p_t.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)

    # Divide by the Sum Norm for both the weak and strong augmentations
    scaler_p_t = check__nans(1 / label_hist).detach()
    modulate_p_t = p_t * scaler_p_t
    modulate_p_t /= modulate_p_t.sum(dim=-1, keepdim=True)

    scaler_prob_s = check__nans(1 / histogram).detach()
    modulate_prob_s = probs_ulb_s.mean(dim=0, keepdim=True) * scaler_prob_s
    modulate_prob_s /= modulate_prob_s.sum(dim=-1, keepdim=True)

    # Cross entropy loss between two Sum Norm logits.
    loss = (modulate_p_t * torch.log(modulate_prob_s + 1e-9)).sum(dim=1).mean()

    return loss, histogram.mean()


def check__nans(x):
    x[x == float('inf')] = 0.0
    return x


def update_para(tau_t, p_t, prob, label_hist, ema_w):
    max_prob, pl = torch.max(prob, dim=1)
    tau_t = ema_w * tau_t + (1 - ema_w) * max_prob.mean()

    a = prob.mean(dim=0)
    p_t = ema_w * p_t + (1 - ema_w) * a

    histogram = torch.bincount(pl, minlength=p_t.shape[0]).to(p_t.dtype)
    label_hist = label_hist * ema_w + (1. - ema_w) * (histogram / histogram.sum())

    return tau_t, p_t, label_hist


def train(model, model1, labeled_trainloader, unlabeled_trainloader, optimizer, optimizer2, ceriation, epoch,
          step, num_classes, class_weights, tau_t, p_t, label_hist, ema_w):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    losses = AverageMeter('Loss', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(labeled_trainloader),
        [batch_time, data_time, losses, top1],
        prefix="Train Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    loss_m, label_m, _, _ = filter_split(model, labeled_trainloader, ceriation, num_classes)
    loss_m_u, label_m_u, _, loss_m_u_all = filter_noisy(model, unlabeled_trainloader, ceriation, num_classes, flag=True)

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = len(labeled_trainloader)
    for batch_idx in range(num_iter):
        try:
            inputs_x, _, targets_x, index = next(labeled_train_iter)
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, _, targets_x, index = next(labeled_train_iter)

        try:
            inputs_u, inputs_u2, target_u, _, index_u = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, target_u, _, index_u = next(unlabeled_train_iter)

        inputs_x, inputs_u, inputs_u2, targets_x, target_u = inputs_x.cuda(), inputs_u.cuda(), inputs_u2.cuda(), targets_x.cuda(), target_u.cuda()

        batch_size = inputs_x.size(0)
        u_bs = inputs_u.size(0)

        inputs_all = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)

        logit_all = model(inputs_all)

        loss_l = F.cross_entropy(logit_all[0:batch_size], targets_x)
        # loss_m[index]=loss_label.detach()
        # loss_l = loss_label

        logit_u1 = logit_all[batch_size:batch_size + u_bs]

        prob_u = F.softmax(model1(inputs_u).detach(), dim=1)

        tau_t, p_t, label_hist = update_para(tau_t, p_t, prob_u, label_hist, ema_w)
        loss_tmp = loss_m_u_all[index_u] * F.one_hot(target_u, num_classes).float()

        loss_m_u[index_u] = torch.sum(loss_tmp, dim=1)
        logit_u2 = logit_all[batch_size + u_bs:batch_size + u_bs * 2]

        # loss_u1 = F.cross_entropy(logit_u1, target_u, reduction='none')
        #
        # loss_r_u1 = regroup_median_split(loss_u1, target_u, step, loss_m, loss_m_u, label_m, index_u).mean()

        loss_u2 = F.cross_entropy(logit_u2, target_u, reduction='none')

        loss_u = regroup_median_split(loss_u2, target_u, step, loss_m, loss_m_u, label_m, index_u).mean()

        loss = (loss_l + loss_u)

        acc1, acc5 = accuracy(logit_all[0:batch_size], targets_x, topk=(1, 5))
        losses.update(loss.item(), inputs_x[0].size(0))
        top1.update(acc1[0], inputs_x[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer2.step()
        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(0)
    return losses.avg, top1.avg.to("cpu", torch.float).item(), tau_t, p_t, label_hist


def update_label(logits, label, b_index, u_ind, tau_t, p_t):
    max_prob, max_target = torch.max(F.softmax(logits.detach(), dim=1), dim=1)
    b = torch.max(p_t, dim=-1)[0]
    tau_t_c = (p_t / b)
    mask = max_prob.ge(tau_t * tau_t_c[max_target])

    for i in range(label.shape[0]):
        if b_index[i] in u_ind and mask[i] == True:
            label[i] = max_target[i]
    return label


# def train_correction(model,model1, train_loader,clean_targets, optimizer, optimizer2,ceriation, epoch, n,num_class,tau_t,p_t,label_hist,ema_w,label_update):
#     batch_time = AverageMeter('Time', ':6.2f')
#     data_time = AverageMeter('Data', ':6.2f')
#     losses = AverageMeter('Loss', ':6.2f')
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     progress = ProgressMeter(
#         len(train_loader),
#         [batch_time, data_time, losses, top1],
#         prefix="Train Epoch: [{}]".format(epoch))
#
#     model.train()
#
#     end = time.time()
#
#     loss_m, label_m, label_p,prob = filter(model,model1, train_loader, ceriation,num_class,flag=True)
#
#
#     pred_consis=torch.eq(label_m,label_p)
#     # pre_l=torch.argmax(label_update,dim=1)
#     # pred_consis=torch.eq(pre_l,label_p)
#     # pred_mconsis=torch.eq(pre_l,label_m)
#
#
#
#     max_prob,pl=torch.max(prob,dim=1)
#     b=torch.max(p_t,dim=-1)[0]
#     tau_t_c=(p_t/b)
#     mask_prob=max_prob.ge(tau_t*tau_t_c[pl])
#
#     mask=mask_prob*pred_consis
#
#     c_ind = torch.nonzero(mask==True).squeeze()
#     u_ind = torch.nonzero(mask== False).squeeze()
#     label_update[u_ind]=F.one_hot(pl.long(),num_class).float()[u_ind]
#
#     n_l=label_m[c_ind]
#     t_l=clean_targets[c_ind]
#     clean_ratio=torch.mean(torch.eq(n_l,t_l).float())
#     print("clean ratio:",clean_ratio)
#
#     tau_t, p_t, label_hist = update_para(tau_t, p_t, prob[c_ind], label_hist, ema_w=ema_w)
#     mom=0.9
#     label_update= mom * label_update+ (1 - mom) * F.softmax(prob.detach(), dim=1)
#
#     for i, (images, labels, index) in enumerate(train_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         images = Variable(images).cuda()
#         labels = Variable(labels).cuda()
#
#         logist = model(images)
#         b_s = logist.size(0)
#         loss_data = ceriation(logist, labels.long())
#
#
#         loss = regroup_median_matrix(loss_data, labels, n, b_s, loss_m, label_m, index, c_ind, u_ind).mean()
#         acc1, acc5 = accuracy(logist, labels, topk=(1, 5))
#         losses.update(loss.item(), images[0].size(0))
#         top1.update(acc1[0], images[0].size(0))
#         # label_update[index]=mom*label_update[index]+(1-mom)*F.softmax(logist.detach(),dim=1)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         optimizer2.step()
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#     progress.display(0)
#     return losses.avg, top1.avg.to("cpu", torch.float).item(),loss_m, label_m, label_p,tau_t,p_t,label_hist,label_update
#
# #
# def train(model,model1, train_loader,clean_targets, optimizer, optimizer2,ceriation, epoch, n,num_class,tau_t,p_t,label_hist,ema_w,label_update):
#     batch_time = AverageMeter('Time', ':6.2f')
#     data_time = AverageMeter('Data', ':6.2f')
#     losses = AverageMeter('Loss', ':6.2f')
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     progress = ProgressMeter(
#         len(train_loader),
#         [batch_time, data_time, losses, top1],
#         prefix="Train Epoch: [{}]".format(epoch))
#
#     model.train()
#
#     end = time.time()
#
#     loss_m, label_m, label_p,prob = filter(model,model1, train_loader, ceriation,num_class,flag=True)
#
#
#     pred_consis=torch.eq(label_m,label_p)
#     # pre_l=torch.argmax(label_update,dim=1)
#     # pred_consis=torch.eq(pre_l,label_p)
#     # pred_mconsis=torch.eq(pre_l,label_m)
#
#
#
#     max_prob,pl=torch.max(prob,dim=1)
#     b=torch.max(p_t,dim=-1)[0]
#     tau_t_c=(p_t/b)
#     mask_prob=max_prob.ge(tau_t*tau_t_c[pl])
#
#     mask=mask_prob*pred_consis
#
#     c_ind = torch.nonzero(mask==True).squeeze()
#     u_ind = torch.nonzero(mask== False).squeeze()
#     label_update[c_ind]=F.one_hot(label_m.long(),num_class).float()[c_ind]
#
#     n_l=label_m[c_ind]
#     t_l=clean_targets[c_ind]
#     clean_ratio=torch.mean(torch.eq(n_l,t_l).float())
#     print("clean ratio:",clean_ratio)
#
#     tau_t, p_t, label_hist = update_para(tau_t, p_t, prob[c_ind], label_hist, ema_w=ema_w)
#     mom=0.9
#     label_update= mom * label_update+ (1 - mom) * F.softmax(prob.detach(), dim=1)
#
#     for i, (images, labels, index) in enumerate(train_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         images = Variable(images).cuda()
#         labels = Variable(labels).cuda()
#
#         logist = model(images)
#         b_s = logist.size(0)
#         loss_data = ceriation(logist, labels.long())
#         loss = regroup_median_matrix(loss_data, labels, n, b_s, loss_m, label_m, index, c_ind, u_ind).mean()
#         acc1, acc5 = accuracy(logist, labels, topk=(1, 5))
#         losses.update(loss.item(), images[0].size(0))
#         top1.update(acc1[0], images[0].size(0))
#         # label_update[index]=mom*label_update[index]+(1-mom)*F.softmax(logist.detach(),dim=1)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         optimizer2.step()
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#     progress.display(0)
#     return losses.avg, top1.avg.to("cpu", torch.float).item(),loss_m, label_m, label_p,tau_t,p_t,label_hist,label_update

def train_warm(model, model1, train_loader, optimizer, optimizer2, ceriation, epoch, n, num_class, label_update):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    losses = AverageMeter('Loss', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Train Epoch: [{}]".format(epoch))

    model.train()
    mom = 0.9
    end = time.time()
    loss_m, label_m, label_p, _ = filter(model, model, train_loader, ceriation, num_class)
    for i, (images, labels, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        label_update[index] = F.one_hot(labels, num_class).float()
        logist = model(images)
        b_s = logist.size(0)
        loss_data = ceriation(logist, labels.long())
        label_update[index] = (1 - mom) * F.softmax(model1(images).detach(), dim=1) + mom * label_update[index]
        loss = loss_data.mean()
        # if epoch <= 2:
        #     loss = loss_data.mean()
        # else:
        #     loss = regroup_median(loss_data, labels, n, b_s, loss_m, label_m, index).mean()
        # # loss=RML_batch(loss_data,n)
        acc1, acc5 = accuracy(logist, labels, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer2.step()
        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(0)
    return losses.avg, top1.avg.to("cpu", torch.float).item(), label_update


def evaluate(model, eva_loader, ceriation, prefix, ignore=-1):
    losses = AverageMeter('Loss', ':3.2f')
    top1 = AverageMeter('Acc@1', ':3.2f')
    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(eva_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            logist = model(images)

            loss = ceriation(logist, labels.long()).mean()
            acc1, acc5 = accuracy(logist, labels, topk=(1, 5))

            losses.update(loss.item(), images[0].size(0))
            top1.update(acc1[0], images[0].size(0))

    if prefix != "":
        print(getTime(), prefix, round(top1.avg.item(), 2))

    return losses.avg, top1.avg.to("cpu", torch.float).item()


def evaluateWithBoth(model1, model2, eva_loader, prefix):
    top1 = AverageMeter('Acc@1', ':3.2f')
    model1.eval()
    model2.eval()

    with torch.no_grad():
        for i, (images, labels, _) in enumerate(eva_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            logist1 = model1(images)
            logist2 = model2(images)
            logist = (F.softmax(logist1, dim=1) + F.softmax(logist2, dim=1)) / 2
            acc1, acc5 = accuracy(logist, labels, topk=(1, 5))
            top1.update(acc1[0], images[0].size(0))

    if prefix != "":
        print(getTime(), prefix, round(top1.avg.item(), 2))

    return top1.avg.to("cpu", torch.float).item()


def predict(predict_loader, model):
    model.eval()
    preds = []
    probs = []

    with torch.no_grad():
        for images, _, _ in predict_loader:
            if torch.cuda.is_available():
                images = Variable(images).cuda()
                logits = model(images)
                outputs = F.softmax(logits, dim=1)
                prob, pred = torch.max(outputs.data, 1)
                preds.append(pred)
                probs.append(prob)

    return torch.cat(preds, dim=0).cpu(), torch.cat(probs, dim=0).cpu()


def predict_softmax(predict_loader, model):
    model.eval()
    softmax_outs = []
    with torch.no_grad():
        for images1, images2, target, _, index in predict_loader:
            if torch.cuda.is_available():
                images1 = Variable(images1).cuda()
                logits1 = model(images1)
                outputs = F.softmax(logits1, dim=1)
                softmax_outs.append(outputs)

    return torch.cat(softmax_outs, dim=0).cpu()
