# coding:utf-8
import os
import argparse
import time
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from util.MF_dataset import MF_dataset
from util.MF_dataset_multilabel import MF_dataset_multilabel
from util.MF_dataset_edge import MF_dataset_edge
from util.util import calculate_accuracy
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise,RandomCrop3,RandomFlip_multilabel,RandomCrop_multilabel
from model import mffenet_multi,mffenetusetrans,mffenet,FuseSegwithmultilabel,fusesegusetrans5,transformerlikefuse,MFNet, SegNet,RTFNet,afnet,SegNet,FuNNet,FuseSeg,afnetusetrans,fusesegusetrans,fusesegusetrans2,fusesegusetrans3,fusesegusetrans4
from model.relight import LightNet
from util.util import get_scheduler
import numpy as np
from tqdm import tqdm
from util.util import probreweighting,FocalLoss
import lovasz_losses as L
import shutil
import torch.nn as nn
best_miou = 0.
is_best = 0
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, input, target):
        loss = 0
        if isinstance(input, list):
            for inp in input:
                # size=inp.size()[2:]
                # print(size)
                # print(target.shape)
                new_target = F.interpolate(target, size=inp.size()[2:], mode='nearest')
                # print(new_target.shape)
                loss += self.cal(inp, new_target)
        else:
            loss = self.cal(input, target)

        return loss

    def cal(self, input, target):
        n, c, h, w = input.size()


        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)

        # print(log_p.shape)
        # print(target_t.shape)
        target_trans = target_t.clone()


        pos_index = (target_t ==1)

        neg_index = (target_t ==0)

        ignore_index=(target_t >1)


        target_trans[pos_index] = 1

        target_trans[neg_index] = 0


        pos_index = pos_index.data.cpu().numpy().astype(bool)

        neg_index = neg_index.data.cpu().numpy().astype(bool)

        ignore_index = ignore_index.data.cpu().numpy().astype(bool)


        weight = torch.Tensor(log_p.size()).fill_(0)

        weight = weight.numpy()

        pos_num = pos_index.sum()

        neg_num = neg_index.sum()

        sum_num = pos_num + neg_num

        weight[pos_index] = neg_num*1.0 / sum_num

        weight[neg_index] = pos_num*1.0 / sum_num


        weight[ignore_index] = 0


        weight = torch.from_numpy(weight)

        weight = weight.cuda()

        # print(log_p.shape)
        # print(target_t.shape)
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        # loss = F.binary_cross_entropy(log_p, target_t, weight, size_average=True)

        return loss
# config
weights=torch.log(torch.FloatTensor([0.91865138 ,0.04526743, 0.01380293 ,0.00426743 ,0.00643076 ,0.00545441,
 0.00150976, 0.00175315, 0.00286275])).cuda()
weights = (torch.mean(weights) - weights) / torch.std(weights) * 0.05 + 1.0
n_class   = 9
# data_dir  = 'E:\google drive/ir_seg_dataset2/'
# model_dir = 'E:\MFNet-pytorch-master/weights/MFNetloss/'
data_dir  = '/content/drive/MyDrive/ir_seg_dataset2'
model_dir = '/content/drive/MyDrive/MFNet-pytorch-master/weights/mffenet_multi/'
augmentation_methods = [
    RandomFlip(prob=0.5),
    # transforms.RandomScale((0.75,1.0,1.25,1.5,1.75, 2.0)),
    RandomCrop(crop_rate=0.2, prob=1.0)



    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]
# lr_start  = 0.01
# lr_decay  = 0.95


def train(epo, model, train_loader, optimizer):

    # lr_this_epo = lr_start * lr_decay**(epo-1)
    for param_group in optimizer.param_groups:

        lr_this_epo=param_group['lr']

    loss_avg = 0.
    acc_avg  = 0.
    start_t = t = time.time()
    cf = np.zeros((n_class, n_class))
    edge_loss = EdgeLoss().cuda()
    model.train()

    for it, (images, labels, boundary_label,binary_label,names) in enumerate(train_loader):
        # images = Variable(images).cuda()
        # labels = Variable(labels).cuda()
        # if args.gpu >= 0:
        images = images.to(args.device)
        labels = labels.to(args.device)
        salient_label = binary_label.to(args.device)
        boundary_label=boundary_label.to(args.device)
        salient_label=salient_label.float()
        boundary_label=boundary_label.float()


        optimizer.zero_grad()
        logits_semantic,logits_boundary,logits_salient = model(images)
        # logits=probreweighting()(logits,labels)
        # logits=logits.to(args.device)
        # labels=labels.to(args.device)
        # loss=FocalLoss(gamma=1,alpha=1)(logits,labels)
        # loss_semantic = F.cross_entropy(logits_semantic, labels,weight=weights)
        semantic_prob=F.softmax(logits_semantic,dim=1)
        loss_semantic=L.lovasz_softmax(semantic_prob,labels,ignore=255)
        # logits_salient=logits_salient.view(-1,480,640).contiguous()
        # logits_boundary=logits_salient.view(-1,480,640).contiguous()
        loss_salient = F.binary_cross_entropy_with_logits(logits_salient,salient_label)
        loss_edge = edge_loss(logits_boundary, boundary_label)
        # loss_boundary = F.binary_cross_entropy_with_logits(logits_boundary, boundary_label)
        loss=loss_semantic+loss_edge+loss_salient
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(logits_semantic, labels)
        loss_avg += float(loss)
        acc_avg  += float(acc)

        cur_t = time.time()
        if cur_t-t > 5:
            print('|- epo %s/%s. train iter %s/%s. %.2f img/sec loss: %.4f,acc: %.4f' \
                % (epo, args.epoch_max, it+1, train_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss),float(acc)))
            t += 5
        predictions = logits_semantic.argmax(1)
        for gtcid in range(n_class):
            for pcid in range(n_class):
                gt_mask = labels == gtcid
                pred_mask = predictions == pcid
                intersection = gt_mask * pred_mask
                cf[gtcid, pcid] += int(intersection.sum())
    IoU = np.diag(cf) / (cf.sum(axis=1) + cf.sum(axis=0) - np.diag(cf))
    Miou = np.mean(IoU)
    content = '| epo:%s/%s lr:%.4f train_loss_avg:%.4f train_acc_avg:%.4f train_miou:%.4f\n' \
              % (epo, args.epoch_max, lr_this_epo, loss_avg / train_loader.n_iter, acc_avg / train_loader.n_iter, Miou)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content)

    # content = '| epo:%s/%s lr:%.4f train_loss_avg:%.4f train_acc_avg:%.4f ' \
    #         % (epo, args.epoch_max, lr_this_epo, loss_avg/train_loader.n_iter, acc_avg/train_loader.n_iter)
    # print(content)
    # with open(log_file, 'a') as appender:
    #     appender.write(content)


def validation(epo, model, val_loader):

    cf = np.zeros((n_class, n_class))
    loss_avg = 0.
    acc_avg  = 0.
    start_t = time.time()
    model.eval()
    edge_loss = EdgeLoss().cuda()

    with torch.no_grad():
        for it, (images, labels,boundary_label,salient_label,names) in enumerate(val_loader):
            # images = Variable(images)
            # labels = Variable(labels)
            # if args.gpu >= 0:
            images = images.to(args.device)
            labels = labels.to(args.device)
            salient_label = salient_label.to(args.device)
            boundary_label = boundary_label.to(args.device)
            salient_label = salient_label.float()
            boundary_label = boundary_label.float()

            logits_semantic,logits_boundary,logits_salient= model(images)
            semantic_prob = F.softmax(logits_semantic, dim=1)
            loss_semantic = L.lovasz_softmax(semantic_prob, labels, ignore=255)
            loss_edge = edge_loss(logits_boundary, boundary_label)
            # logits_salient = logits_salient.view(-1, 480, 640).contiguous()
            # logits_boundary = logits_salient.view(-1, 480, 640).contiguous()
            loss_salient = F.binary_cross_entropy_with_logits(logits_salient, salient_label)
            # loss_boundary = F.binary_cross_entropy_with_logits(logits_boundary, boundary_label)
            loss = loss_semantic+loss_edge+loss_salient
            acc = calculate_accuracy(logits_semantic, labels)
            loss_avg += float(loss)
            acc_avg  += float(acc)

            cur_t = time.time()
            print('|- epo %s/%s. val iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
                    % (epo, args.epoch_max, it+1, val_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss), float(acc)))
            predictions = logits_semantic.argmax(1)
            for gtcid in range(n_class):
                for pcid in range(n_class):
                    gt_mask = labels == gtcid
                    pred_mask = predictions == pcid
                    intersection = gt_mask * pred_mask
                    cf[gtcid, pcid] += int(intersection.sum())
        IoU = np.diag(cf) / (cf.sum(axis=1) + cf.sum(axis=0) - np.diag(cf))
        Miou = np.mean(IoU)
        content = '| val_loss_avg:%.4f val_acc_avg:%.4f val_miou:%.4f\n' \
                  % (loss_avg / val_loader.n_iter, acc_avg / val_loader.n_iter, Miou)
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content)
    # content = '| val_loss_avg:%.4f val_acc_avg:%.4f\n' \
    #         % (loss_avg/val_loader.n_iter, acc_avg/val_loader.n_iter)
    # # acc=acc_avg/val_loader.n_iter
    #
    # print(content)
    # with open(log_file, 'a') as appender:
    #     appender.write(content)
        return Miou


def main():
    random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    mioumax=0.35

    model = eval(args.model_name)(n_class=n_class)
    # if args.gpu >= 0:
    model=model.to(args.device)
    #-------加上lightnet----------------------------------------------------------------------------------
    # lightnet = LightNet()
    # # lightnet.train()
    # lightnet.to(args.device)
    # -------加上lightnet----------------------------------------------------------------------------------
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, gamma=args.lr_decay, last_epoch=-1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
    lr_scheduler = get_scheduler(optimizer, args)

    if args.epoch_from > 1:
        print('| loading checkpoint file %s... ' % checkpoint_model_file, end='')
        model.load_state_dict(torch.load(checkpoint_model_file))
        optimizer.load_state_dict(torch.load(checkpoint_optim_file))
        print('done!')

    train_dataset = MF_dataset_edge(data_dir, 'train', have_label=True, transform=augmentation_methods)
    val_dataset  = MF_dataset_edge(data_dir, 'test', have_label=True)

    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True
    )
    val_loader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    train_loader.n_iter = len(train_loader)
    val_loader.n_iter   = len(val_loader)
    miou = 0

    for epo in tqdm(range(args.epoch_from, args.epoch_max + 1)):
        print('\n| epo #%s begin...' % epo)
        global best_miou, is_best

        train(epo, model, train_loader, optimizer)
        lr_scheduler.step()
        # lr_scheduler.step()
        # miou=validation(epo, model, val_loader)
        if epo > 30 and epo % 2 == 0:
            miou = validation(epo, model, val_loader)
        elif epo <= 30 and epo % 5 == 0:
            miou = validation(epo, model, val_loader)
            # torch.save(model.state_dict(), best_model_file)
            # torch.save(edgenet.state_dict(), edge_best_model_file)
        # lr_scheduler.step()
        is_best = miou > best_miou
        torch.save(model.state_dict(), checkpoint_model_file)
        torch.save(optimizer.state_dict(), checkpoint_optim_file)
        print('done!')

        if is_best:
            best_miou = miou
            shutil.copy(checkpoint_model_file, best_model_file)
            is_best = 0

        # save check point model
        # print('| saving check point model file... ', end='')
        # torch.save(model.state_dict(), checkpoint_model_file)
        # # torch.save(edgenet.state_dict(), edge_checkpoint_model_file)
        # torch.save(optimizer.state_dict(), checkpoint_optim_file)
        # torch.save(optimizer_edge.state_dict(), edge_checkpoint_optim_file)
        # torch.save(model.state_dict(), checkpoint_model_file)
        if epo == 200:
            torch.save(model.state_dict(), final_model_file)

        # print('done!')

        # lr_scheduler_edge.step()

    # os.rename(checkpoint_model_file, final_model_file)
    # os.rename(edge_checkpoint_model_file, edge_final_model_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train MFNet with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='mffenet_multi')
    parser.add_argument('--batch_size',  '-B',  type=int, default=4)
    parser.add_argument('--epoch_max' ,  '-E',  type=int, default=100)
    parser.add_argument('--epoch_from',  '-EF', type=int, default=1)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=1)
    parser.add_argument('--lr_power', default=0.9, type=float)
    parser.add_argument('--lr_policy', default='poly', type=str)
    parser.add_argument('--lr_start', '-ls', type=float, default=0.01)
    parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    args = parser.parse_args()

    model_dir = os.path.join(model_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_model_file = os.path.join(model_dir, 'tmp.pth')
    checkpoint_optim_file = os.path.join(model_dir, 'tmp.optim')
    final_model_file      = os.path.join(model_dir, 'final.pth')
    log_file              = os.path.join(model_dir, 'log.txt')
    best_model_file       = os.path.join(model_dir, 'best.pth')

    print('| training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('| from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('| model will be saved in: %s' % model_dir)

    main()
