import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import torch.nn.functional as F
from torch.autograd import Variable


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


def Median_filter(input, target):
    input = -F.log_softmax(input, dim=-1)
    loss_v = torch.sum(input * target, dim=1)
    bs = input.size(0)

    for i in range(bs):
        index = np.random.randint(0, bs, 5)
        s_loss = loss_v[index]
        loss_v[i] = s_loss.median()
    return loss_v.mean()


# def regroup_median_matrix(loss_data, label, step,n, b_s, loss_m, label_m, s_index):
#     loss_v = loss_data
#     loss = []
#
#     for i in range(b_s):
#         index = torch.nonzero(label_m == label[i]).squeeze()
#
#         loss_x = loss_m[index]
#
#         loss_norm = torch.exp(-loss_x ** 2 - loss_x)
#         loss_norm = loss_norm / (loss_norm.sum() + 0.00000001)
#         # print(loss_norm)
#         loss_x_i_1 = loss_m[s_index[i]]
#         select_index = torch.multinomial(loss_norm, step * n, replacement=False)
#         select_loss = loss_x[select_index].detach()
#         select_loss = select_loss.view(n, step)
#         # select_loss_median,_ = torch.median(select_loss, dim=1)
#         select_loss_mean = torch.mean(select_loss, dim=1)
#         select_loss = torch.cat([loss_x_i_1.view(1), select_loss_mean])
#         select_loss = torch.median(select_loss)
#         l_weight = select_loss / (loss_x_i_1 + 0.0000001)
#         if loss_v[i].detach() <= loss_v[i].detach() * l_weight or loss_v[i] == 0:
#             loss.append(loss_v[i].view(1))
#
#         else:
#             tmp = l_weight * loss_v[i]
#             loss.append(tmp.view(1))
#     loss = torch.cat(loss, dim=0)
#     return loss

def regroup_median_matrix(loss_data, label,n, b_s, loss_m, label_m, s_index, c_ind, u_ind):
    loss_v = loss_data
    loss = []
    label_m_i = label_m[c_ind]
    loss_m_i = loss_m[c_ind]
    for i in range(b_s):

        if s_index[i] in u_ind:
            index = torch.nonzero(label_m_i == label[i]).squeeze()
            loss_x = loss_m_i[index]
            e_num = (int(loss_x.size(0) / n)+1)*n-loss_x.size(0)
            if e_num<=(n/2):
                if e_num > 0:
                    r_idx = torch.randint(0, loss_x.size(0), [loss_x.size(0)]).to(label.device)
                    loss_x = loss_x[r_idx]
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
                if loss_v[i].detach() <= loss_v[i].detach() * l_weight or loss_v[i] == 0:
                    loss.append(loss_v[i].view(1))

                else:
                    tmp = l_weight * loss_v[i]
                    loss.append(tmp.view(1))
            else:
                loss_data[i]=0.
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

        if e_num > 0:
            r_idx = torch.randint(0, loss_x.size(0), [loss_x.size(0)]).to(label.device)
            loss_x = loss_x[r_idx]
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
        if loss_v[i].detach() <= loss_v[i].detach() * l_weight or loss_v[i] == 0:
                    loss.append(loss_v[i].view(1))

        else:
            tmp = l_weight * loss_v[i]
            loss.append(tmp.view(1))

    loss = torch.cat(loss, dim=0)
    return loss

def RML_batch(loss_v,n):

    bs=loss_v.size(0)
    e_num = (int(bs / n) + 1) * n - bs
    if e_num>0:
        r_idx = torch.randint(0, bs, [bs]).to(loss_v.device)
        loss_x = loss_v[r_idx]
        loss_norm = torch.exp(-loss_x ** 2 - loss_x)
        loss_norm = loss_norm / (loss_norm.sum() + 0.00000001)
        select_index = torch.multinomial(loss_norm, e_num, replacement=False)
        loss_v=torch.cat([loss_v,loss_v[select_index]],dim=0)
    loss_v=loss_v.view(n,int(bs/n)+1)
    loss_v=torch.mean(loss_v,dim=1)
    return torch.median(loss_v)



def obtain_c_soft(center, feature):
    batch_size = feature.size(0)
    center = center.unsqueeze(0).repeat(batch_size, 1, 1)
    c_feature = feature.unsqueeze(1).repeat(1, feature.size(1), 1)
    distance = torch.sum((c_feature - center) ** 2, dim=-1)
    soft_target = F.softmax(-distance,dim=1)
    return soft_target
def filter_correction(model, train_loader,class_center,label_update,lam,c_ind,u_ind):
    label_m = torch.zeros(50000).cuda()
    label_p = torch.zeros(50000).cuda()
    loss_m = torch.zeros(50000).cuda()
    logits= torch.zeros(50000,class_center.size(0)).cuda()
    with torch.no_grad():
        for i, (images, labels, index) in enumerate(train_loader):
            images,labels= images.cuda(),labels.cuda()

            output = model(images)
            targets=label_update[index]
            loss_m[index] = -torch.sum(torch.log_softmax(output, dim=1) * targets, dim=1)
            label_m[index]=labels.float()
            pred_label = torch.argmax(torch.softmax(output, dim=1), dim=1)
            label_p[index] = pred_label.float()
            logits[index]=output.detach()


    label_c = torch.argmax(label_update, dim=1)

    prob_all=F.softmax(logits,dim=1)

    # class_center=update_center(prob_all[c_ind],label_c[c_ind],class_center,lam)
    # soft_pred=obtain_c_soft(class_center,prob_all)
    #
    max_probs, targets_u = torch.max(prob_all, dim=-1)
    # prob=torch.sum(prob_all*F.one_hot(label_m.long(),class_center.size(0)).float(),dim=1)[u_ind]
    u_consis=max_probs.ge(0.95)
    s_u_ind=torch.nonzero(u_consis == True).squeeze()
    if len(s_u_ind.size())==0:
        update_ind=u_ind[s_u_ind]

        new_pred=F.one_hot(targets_u.long(),class_center.size(0))
        # omega=(loss_m-loss_m.min())/(loss_m.max()-loss_m.min())
        # label_update[u_ind]=omega[u_ind].view(-1,1)*label_update[u_ind]+(1-omega[u_ind].view(-1,1))*soft_pred
        label_update[update_ind]=new_pred.float()[update_ind]
        loss_m[update_ind]=-torch.log(torch.sum(new_pred*label_update,dim=1))[update_ind]
    return loss_m, label_m, label_update,c_ind,u_ind,class_center

def update_center(logits,label_m,class_center,lam):

    for i in range(logits.size(1)):
        index_i = torch.nonzero(label_m == i)
        feature_i = logits[index_i]
        feature_i = torch.mean(feature_i, dim=0, keepdim=True)

        class_center[i:i + 1] = (1 - lam) * feature_i + lam * class_center[i:i + 1]

    class_center = F.normalize(class_center, dim=1, p=2)
    # class_center = torch.log_softmax(class_center, dim=1)
    return class_center
def train_correction(model,model1, train_loader, optimizer, optimizer2,epoch, n,class_center,lam,label_update,loss_m,label_m,c_ind,u_ind):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    losses = AverageMeter('Loss', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Train Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()

    loss_m, label_m, label_update,c_ind,u_ind,class_center = filter_correction(model1, train_loader,class_center,label_update,lam,c_ind,u_ind)
    label_c=torch.argmax(label_update,dim=1)

    for i, (images, _, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        labels=label_c[index]
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        targets=label_update[index]
        logist = model(images)
        b_s = logist.size(0)
        loss_data = -torch.sum(torch.log_softmax(logist, dim=1) * targets, dim=1)
        loss = regroup_median_matrix(loss_data, labels, n, b_s, loss_m, label_c, index, c_ind, u_ind).mean()
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
    return losses.avg, top1.avg.to("cpu", torch.float).item(),class_center,label_update,loss_m,label_m,c_ind,u_ind

def filter(model, train_loader, ceriation,num_class):
    label_m = torch.zeros(50000).cuda()
    label_p = torch.zeros(50000).cuda()
    loss_m = torch.zeros(50000).cuda()
    logits= torch.zeros(50000,num_class).cuda()
    with torch.no_grad():
        for i, (images, labels, index) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            if labels.dim() > 1:
                loss_m[index] = -torch.mean(torch.sum(torch.log_softmax(output, dim=1) * labels, dim=1), dim=0).float()
            else:
                loss_m[index] = ceriation(output, labels.long())
            label_m[index]=labels.float()
            pred_label = torch.argmax(torch.softmax(output, dim=1), dim=1)
            label_p[index] = pred_label.float()
            logits[index]=F.softmax(output,dim=1)

    # label_m = torch.cat(label_m, dim=0)
    # label_p = torch.cat(label_p, dim=0)
    return loss_m, label_m, label_p
def train(model,model1, train_loader, optimizer, optimizer2,ceriation, epoch, n,num_class,c_ind,u_ind):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    losses = AverageMeter('Loss', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Train Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()

    loss_m, label_m, label_p = filter(model1, train_loader, ceriation,num_class)
    if epoch%1==0:
        pred_consis=torch.eq(label_m,label_p)
        c_ind = torch.nonzero(pred_consis==True).squeeze()
        u_ind = torch.nonzero(pred_consis == False).squeeze()
    for i, (images, labels, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        logist = model(images)
        b_s = logist.size(0)
        loss_data = ceriation(logist, labels.long())
        loss = regroup_median_matrix(loss_data, labels, n, b_s, loss_m, label_m, index, c_ind, u_ind).mean()
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
    return losses.avg, top1.avg.to("cpu", torch.float).item(),loss_m, label_m, label_p,c_ind,u_ind
def train_warm(model, train_loader, optimizer, optimizer2, ceriation, epoch, n,num_class,label_update):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    losses = AverageMeter('Loss', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Train Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    loss_m, label_m, label_p = filter(model, train_loader, ceriation,num_class)
    for i, (images, labels, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        label_update[index]=F.one_hot(labels,num_class).float()
        logist = model(images)
        b_s = logist.size(0)
        loss_data = ceriation(logist, labels.long())
        loss = regroup_median(loss_data,labels,n,b_s,loss_m,label_m,index).mean()
        # loss=RML_batch(loss_data,n)
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
    return losses.avg, top1.avg.to("cpu", torch.float).item(),label_update

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
        for images1, images2 in predict_loader:
            if torch.cuda.is_available():
                images1 = Variable(images1).cuda()
                images2 = Variable(images2).cuda()
                logits1 = model(images1)
                logits2 = model(images2)
                outputs = (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2
                softmax_outs.append(outputs)

    return torch.cat(softmax_outs, dim=0).cpu()
