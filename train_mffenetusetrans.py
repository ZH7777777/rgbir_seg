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

from util.util import calculate_accuracy
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise,RandomCrop3,RandomFlip_multilabel,RandomCrop_multilabel
from model import mffenetusetrans,mffenet,FuseSegwithmultilabel,fusesegusetrans5,transformerlikefuse,MFNet, SegNet,RTFNet,afnet,SegNet,FuNNet,FuseSeg,afnetusetrans,fusesegusetrans,fusesegusetrans2,fusesegusetrans3,fusesegusetrans4
from model.relight import LightNet
from util.util import get_scheduler
import numpy as np
from tqdm import tqdm
from util.util import probreweighting,FocalLoss
import lovasz_losses as L
import torch.nn as nn

# config
weights=torch.log(torch.FloatTensor([0.91865138 ,0.04526743, 0.01380293 ,0.00426743 ,0.00643076 ,0.00545441,
 0.00150976, 0.00175315, 0.00286275])).cuda()
weights = (torch.mean(weights) - weights) / torch.std(weights) * 0.05 + 1.0
n_class   = 9
data_dir  = 'E:\google drive/ir_seg_dataset2/'
model_dir = 'E:\MFNet-pytorch-master/weights/MFNetloss/'
# data_dir  = '/content/drive/MyDrive/ir_seg_dataset2'
# model_dir = '/content/drive/MyDrive/MFNet-pytorch-master/weights/mffenetusetrans/'
augmentation_methods = [
    RandomFlip_multilabel(prob=0.5),
    RandomCrop_multilabel(crop_rate=0.1,prob=1.0)



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
    model.train()

    for it, (images, labels, salient_label,boundary_label,names) in enumerate(train_loader):
        # images = Variable(images).cuda()
        # labels = Variable(labels).cuda()
        # if args.gpu >= 0:
        images = images.to(args.device)
        labels = labels.to(args.device)
        salient_label = salient_label.to(args.device)
        boundary_label=boundary_label.to(args.device)
        salient_label=salient_label.float()
        boundary_label=boundary_label.float()


        optimizer.zero_grad()
        logits_semantic,logits_salient,logits_boundary = model(images)
        # logits=probreweighting()(logits,labels)
        # logits=logits.to(args.device)
        # labels=labels.to(args.device)
        # loss=FocalLoss(gamma=1,alpha=1)(logits,labels)
        loss_semantic = F.cross_entropy(logits_semantic, labels,weight=weights)
        # semantic_prob=F.softmax(logits_semantic,dim=1)
        # loss_semantic=L.lovasz_softmax(semantic_prob,labels,ignore=255)
        # logits_salient=logits_salient.view(-1,480,640).contiguous()
        # logits_boundary=logits_salient.view(-1,480,640).contiguous()
        # loss_salient = F.binary_cross_entropy_with_logits(logits_salient,salient_label)
        # loss_boundary = F.binary_cross_entropy_with_logits(logits_boundary, boundary_label)
        loss=loss_semantic
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

    with torch.no_grad():
        for it, (images, labels,salient_label,boundary_label,names) in enumerate(val_loader):
            # images = Variable(images)
            # labels = Variable(labels)
            # if args.gpu >= 0:
            images = images.to(args.device)
            labels = labels.to(args.device)
            salient_label = salient_label.to(args.device)
            boundary_label = boundary_label.to(args.device)
            salient_label = salient_label.float()
            boundary_label = boundary_label.float()

            logits_semantic,logits_salient,logits_boundary = model(images)
            semantic_prob = F.softmax(logits_semantic, dim=1)
            loss_semantic = L.lovasz_softmax(semantic_prob, labels, ignore=255)
            # logits_salient = logits_salient.view(-1, 480, 640).contiguous()
            # logits_boundary = logits_salient.view(-1, 480, 640).contiguous()
            # loss_salient = F.binary_cross_entropy_with_logits(logits_salient, salient_label)
            # loss_boundary = F.binary_cross_entropy_with_logits(logits_boundary, boundary_label)
            loss = loss_semantic
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
def _reset_parameters(a,b,c,d,e,f,g,h):
    nn.init.xavier_uniform_(a)
    nn.init.xavier_uniform_(b)
    nn.init.xavier_uniform_(c)
    nn.init.xavier_uniform_(d)
    nn.init.xavier_uniform_(e)
    nn.init.xavier_uniform_(f)
    nn.init.xavier_uniform_(g)
    nn.init.xavier_uniform_(h)

def main():
    random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    mioumax=0.35
    external_seq_rgb_1 = nn.Parameter(torch.Tensor(10, 64))
    external_seq_ir_1 = nn.Parameter(torch.Tensor(10, 64))
    external_seq_rgb_2 = nn.Parameter(torch.Tensor(10, 128))
    external_seq_ir_2= nn.Parameter(torch.Tensor(10, 128))
    external_seq_rgb_3 = nn.Parameter(torch.Tensor(10, 256))
    external_seq_ir_3= nn.Parameter(torch.Tensor(10, 256))
    external_seq_rgb_4 = nn.Parameter(torch.Tensor(10, 512))
    external_seq_ir_4= nn.Parameter(torch.Tensor(10, 512))
    _reset_parameters(external_seq_rgb_1,external_seq_ir_1,external_seq_rgb_2,external_seq_ir_2,external_seq_rgb_3,external_seq_ir_3,external_seq_rgb_4,external_seq_ir_4)
    batch_external_seq_rgb_1 = external_seq_rgb_1.repeat(args.batch_size, 1, 1)
    batch_external_seq_ir_1 = external_seq_ir_1.repeat(args.batch_size, 1, 1)
    batch_external_seq_rgb_2 = external_seq_rgb_2.repeat(args.batch_size, 1, 1)
    batch_external_seq_ir_2 = external_seq_ir_2.repeat(args.batch_size, 1, 1)
    batch_external_seq_rgb_3 = external_seq_rgb_3.repeat(args.batch_size, 1, 1)
    batch_external_seq_ir_3 = external_seq_ir_3.repeat(args.batch_size, 1, 1)
    batch_external_seq_rgb_4 = external_seq_rgb_4.repeat(args.batch_size, 1, 1)
    batch_external_seq_ir_4 = external_seq_ir_4.repeat(args.batch_size, 1, 1)
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

    train_dataset = MF_dataset_multilabel(data_dir, 'train', have_label=True, transform=augmentation_methods)
    val_dataset  = MF_dataset_multilabel(data_dir, 'val', have_label=True)

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

    for epo in tqdm(range(args.epoch_from, args.epoch_max+1)):
        print('\n| epo #%s begin...' % epo)

        train(epo, model, train_loader, optimizer)
        # lr_scheduler.step()
        miou=validation(epo, model, val_loader)
        if miou>=mioumax and epo>30:
            mioumax=miou
            torch.save(model.state_dict(), best_model_file)
        # lr_scheduler.step()

        # save check point model
        print('| saving check point model file... ', end='')
        torch.save(model.state_dict(), checkpoint_model_file)
        torch.save(optimizer.state_dict(), checkpoint_optim_file)
        print('done!')
        lr_scheduler.step()

    os.rename(checkpoint_model_file, final_model_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train MFNet with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='mffenetusetrans')
    parser.add_argument('--batch_size',  '-B',  type=int, default=2)
    parser.add_argument('--epoch_max' ,  '-E',  type=int, default=100)
    parser.add_argument('--epoch_from',  '-EF', type=int, default=1)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=1)
    parser.add_argument('--lr_power', default=0.9, type=float)
    parser.add_argument('--lr_policy', default='poly', type=str)
    parser.add_argument('--lr_start', '-ls', type=float, default=0.01)
    parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
    parser.add_argument('--device', default='cpu',
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
