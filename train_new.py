# coding:utf-8
import os
import argparse
import time
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import shutil

from util.Edge_dateset import Edge_dataset
from util.MF_dataset_new import MF_dataset_new
from util.B_dataset import B_dataset
from util.MF_dataset import MF_dataset
from util.MF_dataset_reconstruct import MF_dataset_reconstruct
from util.util import calculate_accuracy

from model import mffenetusetrans,mffenet,mffenetusetrans_cross,mffenetusetrans_c,mffenetusecswintrans,fusesegusetrans_new,fusesegusetrans5,transformerlikefuse,MFNet, SegNet,RTFNet,afnet,SegNet,FuNNet,FuseSeg,afnetusetrans,fusesegusetrans,fusesegusetrans2,fusesegusetrans3,fusesegusetrans4

from util.util import get_scheduler
import numpy as np
from tqdm import tqdm
from util.util import probreweighting,FocalLoss
import lovasz_losses as L
from util.util import runningScore
miou=0
best_miou = 0.
is_best = 0
# config
weights=torch.log(torch.FloatTensor([0.91865138 ,0.04526743, 0.01380293 ,0.00426743 ,0.00643076 ,0.00545441,
 0.00150976, 0.00175315, 0.00286275])).cuda()
weights = (torch.mean(weights) - weights) / torch.std(weights) * 0.05 + 1.0
n_class   = 9
data_dir  = 'E:\google drive/ir_seg_dataset2/'
model_dir = 'E:\MFNet-pytorch-master/weights/fusesegusetrans_lovaz/'
# data_dir='/content/drive/MyDrive/edgenet/datasets/B(541)'
# data_dir  = '/content/drive/MyDrive/ir_seg_dataset2'
# model_dir = '/content/drive/MyDrive/MFNet-putorch-master/weights/mffenet_new/'




def train(epo, model, train_loader, optimizer,running_metrics):

    # lr_this_epo = lr_start * lr_decay**(epo-1)
    for param_group in optimizer.param_groups:

        lr_this_epo=param_group['lr']
    print('lr is:%.4f',lr_this_epo)

    loss_avg = 0.
    acc_avg  = 0.
    start_t = t = time.time()
    cf = np.zeros((n_class, n_class))
    model.train()

    for it, (images, labels, names) in enumerate(train_loader):
        # images = Variable(images).cuda()
        # labels = Variable(labels).cuda()
        # if args.gpu >= 0:
        images = images.to(args.device)
        labels = labels.to(args.device)

        optimizer.zero_grad()
        logits = model(images)
        # logits=probreweighting()(logits,labels)
        # logits=logits.to(args.device)
        # labels=labels.to(args.device)
        # loss=FocalLoss(gamma=1,alpha=1)(logits,labels)
        # loss = F.cross_entropy(logits, labels,weight=weights)
        pred=F.softmax(logits,dim=1)
        loss=L.lovasz_softmax(pred,labels)
        running_metrics.update(labels.cpu().numpy(), logits.argmax(1).cpu().numpy())
        # metric_scores = {'iou': running_metrics.get_scores()[0]["Mean IoU : \t"]}
        # a=running_metrics.get_scores()
        # result = running_metrics.get_scores()
        # IoU = result[1]
        # Miou = result[0]["Mean IoU : \t"]
        # pixel_acc = result[0]["Overall Acc: \t"]
        # recall = result[0]["Mean Acc/recall : \t"]
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(logits, labels)
        loss_avg += float(loss)
        acc_avg  += float(acc)

        cur_t = time.time()
        if cur_t-t > 5:
            print('|- epo %s/%s. train iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
                % (epo, args.epoch_max, it+1, train_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss), float(acc)))
            t += 5
        predictions = logits.argmax(1)
        # for gtcid in range(n_class):
        #     for pcid in range(n_class):
        #         gt_mask = labels == gtcid
        #         pred_mask = predictions == pcid
        #         intersection = gt_mask * pred_mask
        #         cf[gtcid, pcid] += int(intersection.sum())
    result=running_metrics.get_scores()
    IoU = result[1]
    Miou = result[0]["Mean IoU : \t"]
    pixel_acc=result[0]["Overall Acc: \t"]
    recall=result[0]["Mean Acc/recall : \t"]
    content = '| epo:%s/%s lr:%.4f train_loss_avg:%.4f train_pixel_acc:%.4f train_recall_acc:%.4f train_miou:%.4f \n' \
              % (epo, args.epoch_max, lr_this_epo, loss_avg / train_loader.n_iter, pixel_acc,recall, Miou)
    print(content)
    print("|class iou:")
    print(IoU)

    with open(log_file, 'a') as appender:
        appender.write(content)

    # content = '| epo:%s/%s lr:%.4f train_loss_avg:%.4f train_acc_avg:%.4f ' \
    #         % (epo, args.epoch_max, lr_this_epo, loss_avg/train_loader.n_iter, acc_avg/train_loader.n_iter)
    # print(content)
    # with open(log_file, 'a') as appender:
    #     appender.write(content)


def validation(epo, model, val_loader,running_metrics):

    cf = np.zeros((n_class, n_class))
    loss_avg = 0.
    acc_avg  = 0.
    start_t = time.time()
    model.eval()
    running_metrics.reset()

    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            # images = Variable(images)
            # labels = Variable(labels)
            # if args.gpu >= 0:
            images = images.to(args.device)
            labels = labels.to(args.device)

            logits = model(images)
            running_metrics.update(labels.cpu().numpy(), logits.argmax(1).cpu().numpy())
            pred = F.softmax(logits, dim=1)
            loss = L.lovasz_softmax(pred, labels)
            acc = calculate_accuracy(logits, labels)
            loss_avg += float(loss)
            acc_avg  += float(acc)

            cur_t = time.time()
            print('|- epo %s/%s. val iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
                    % (epo, args.epoch_max, it+1, val_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss), float(acc)))
            predictions = logits.argmax(1)
            for gtcid in range(n_class):
                for pcid in range(n_class):
                    gt_mask = labels == gtcid
                    pred_mask = predictions == pcid
                    intersection = gt_mask * pred_mask
                    cf[gtcid, pcid] += int(intersection.sum())
        result = running_metrics.get_scores()
        IoU = result[1]
        Miou = result[0]["Mean IoU : \t"]
        pixel_acc = result[0]["Overall Acc: \t"]
        recall = result[0]["Mean Acc/recall : \t"]
        content = '| val_loss_avg:%.4f val_pixel_acc:%.4f val_recall_acc:%.4f val_miou:%.4f\n' \
                  % (loss_avg / val_loader.n_iter, pixel_acc,recall, Miou)
        print(content)
        print("|class iou:")
        print(IoU)
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
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    cropsize = [384, 288]
    randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)
    train_dataset = MF_dataset_reconstruct(data_dir, cropsize=cropsize, mode='train', randomscale=randomscale)
    val_dataset  = MF_dataset(data_dir, mode='test', randomscale=randomscale)

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
    miou=0
    running_metrics=runningScore(n_class)

    for epo in tqdm(range(args.epoch_from, args.epoch_max+1)):
        print('\n| epo #%s begin...' % epo)
        global best_miou, is_best

        train(epo, model, train_loader, optimizer,running_metrics)
        lr_scheduler.step()
        if epo > 30 and epo % 2 == 0:
            miou = validation(epo, model, val_loader,running_metrics)
        elif epo <= 30 and epo % 5 == 0:
            miou = validation(epo, model, val_loader,running_metrics)
        # lr_scheduler.step()

        # save check point model
        running_metrics.reset()
        is_best = miou > best_miou
        torch.save(model.state_dict(), checkpoint_model_file)
        torch.save(optimizer.state_dict(), checkpoint_optim_file)
        print('done!')

        if is_best:
            best_miou = miou
            shutil.copy(checkpoint_model_file, best_model_file)
            is_best = 0

    # os.rename(checkpoint_model_file, final_model_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train MFNet with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='mffenet')
    parser.add_argument('--batch_size',  '-B',  type=int, default=4)
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

    parser.add_argument('--warmup_epochs', '-warmup_epochs', type=float, default=10)
    parser.add_argument('--warmup_start_lr', '-warmup_start_lr', type=float, default=1e-5)
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
