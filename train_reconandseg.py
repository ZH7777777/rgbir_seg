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
from util.augmentation import ConvertFromInts, RandomBrightness,RandomContrast,RandomSaturation,RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise,RandomCrop3,RandomFlip_multilabel,RandomCrop_multilabel
from model import DenseFuse_net,SRDenseNet,FuseSeg_cat,backbonedensenet,mffenet_vgg,pool_mffenet,ccaffmnet,mffenetusetrans,mffenet,mffenetusetrans_cross,mffenetusetrans_c,mffenetusecswintrans,fusesegusetrans_new,fusesegusetrans5,transformerlikefuse,MFNet, SegNet,RTFNet,afnet,SegNet,FuNNet,FuseSeg,afnetusetrans,fusesegusetrans,fusesegusetrans2,fusesegusetrans3,fusesegusetrans4
from model.relight import LightNet
from util.util import get_scheduler
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torchvision
from util.util import probreweighting,FocalLoss
import lovasz_losses as L

miou=0
from model.vgg16_c import VGG16_C
best_miou = 0.
is_best = 0
# config
weights=torch.log(torch.FloatTensor([0.91865138 ,0.04526743, 0.01380293 ,0.00426743 ,0.00643076 ,0.00545441,
 0.00150976, 0.00175315, 0.00286275])).cuda()
weights = (torch.mean(weights) - weights) / torch.std(weights) * 0.05 + 1.0
n_class   = 9
# data_dir  = 'E:\google drive/ir_seg_dataset2/'
# model_dir = 'E:\MFNet-pytorch-master/weights/fusesegusetrans_lovaz/'
from ssim import SSIM
import math
from audtorch.metrics.functional import pearsonr
# data_dir='/content/drive/MyDrive/edgenet/datasets/B(541)'
data_dir  = '/content/drive/MyDrive/ir_seg_dataset2'
# data_dir='/content/drive/MyDrive/RoadScene/'
model_dir = '/content/drive/MyDrive/MFNet-pytorch-master/weights/mffenet_reconrgbir/'
# data_dir='E:\google drive\RoadScene'
# model_dir = 'E:\MFNet-pytorch-master/weights/fusesegusetrans_lovaz/'
def PSNR(target, ref):
    target = np.array(target, dtype=np.float32)
    ref = np.array(ref, dtype=np.float32)

    if target.shape != ref.shape:
        raise ValueError('输入图像的大小应该一致！')

    diff = ref - target
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))
    psnr = 20 * math.log10(255 / rmse)

    return psnr
augmentation_methods = [
    # ConvertFromInts(),
    # RandomBrightness(),
    # RandomContrast(),
    # RandomSaturation(),
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1,prob=1.0),



    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]
# lr_start  = 0.01
# lr_decay  = 0.95

class grad(nn.Module):
    def __init__(self):
        super(grad, self).__init__()
        lap = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
        lap = torch.FloatTensor(lap).unsqueeze(0).unsqueeze(0)
        self.lap = nn.Parameter(lap, requires_grad=False)


    def forward(self, x):
        b, c, h, w = x[0].size()
        grad_list=torch.zeros((b)).cuda()
        k=len(x)
        for i in range(len(x)):
            b, c, h, w = x[i].size()
            kernel = torch.repeat_interleave(self.lap, dim=0, repeats=c)
            kernel = torch.repeat_interleave(kernel, dim=1, repeats=c).cuda()

            grad = F.conv2d(x[i], kernel, padding=1)
            grad2 = torch.square(torch.abs(grad))
            # a = torch.sum(torch.square(grad2))
            a=torch.sum(grad2,dim=[1,2,3])
            # a = torch.sum(a, dim=2)
            # a = torch.sum(a, dim=1)
            # grad3 = torch.sum(torch.square(grad2)) / (c * h * w)
            grad3 = a / (c * h * w)
            grad_list=grad_list+grad3



        return grad_list/len(x)
def train(epo, model_recon,model_seg,rgbbackbone,irbackbone, train_loader, optimizer,rgb_gradloss,ir_gradloss,c):

    # lr_this_epo = lr_start * lr_decay**(epo-1)
    for param_group in optimizer.param_groups:

        lr_this_epo=param_group['lr']

    loss_avg = 0.
    acc_avg  = 0.
    start_t = t = time.time()
    cf = np.zeros((n_class, n_class))
    model_recon.train()
    model_seg.train()
    rgbbackbone.train()
    irbackbone.train()
    ssimloss=SSIM(window_size=11)
    mse=torch.nn.MSELoss(reduce=True, size_average=True)

    # for it, (images, labels, names) in enumerate(train_loader):
    # for it, (images_yir,images, labels, names) in enumerate(train_loader):
    # for it, (images_yir,y_label,ir_label, names) in enumerate(train_loader):
    for it, (images_yir, labels, names) in enumerate(train_loader):
        # images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        # if args.gpu >= 0:
        images_yir = images_yir.to(args.device)
        # print(images_yir.shape)
        # y_label = y_label.to(args.device)
        # ir_label = ir_label.to(args.device)
        # images = images.to(args.device)
        # labels = labels.to(args.device)
        y=torch.cat((images_yir[:, :1], images_yir[:, :1], images_yir[:, :1]), dim=1)
        # rgb=images[:, :3]
        ir = torch.cat((images_yir[:, 3:], images_yir[:, 3:], images_yir[:, 3:]), dim=1)
        cb=images_yir[:,1].unsqueeze(1)
        cr = images_yir[:, 2].unsqueeze(1)
        # rgb=images_yir[:,:3]
        rgb_backbone_feature=rgbbackbone(y)
        ir_backbone_feature=irbackbone(ir)
        y_1=images_yir[:,:1]
        ir_1 = images_yir[:, 3:]
        reconinput=torch.cat((y_1,ir_1),dim=1)

        optimizer.zero_grad()
        # logits,reconstruct= model(images)
        #------------------------densefuse----------------------
        # en=model.encoder(images_yir)
        # reconstruct=model.decoder(en)

        recon = model_recon(reconinput)
        seginput=torch.cat((recon,cb,cr),dim=1)
        logits=model_seg(seginput)
        # print(reconstruct.shape)
        # logits = model(images_yir)
        ##---------------------------融合loss-----------------------------
        rgb_grad=rgb_gradloss(rgb_backbone_feature)/3000
        ir_grad=ir_gradloss(ir_backbone_feature)/3000

        w=torch.softmax(torch.cat((torch.unsqueeze(rgb_grad,dim=-1), torch.unsqueeze(ir_grad,dim=-1)),dim=-1),dim=-1)
        # print(w.shape)
        # print(w[:,0])
        # print(w[:, 1])

        SSIM_rgb = 1 - ssimloss(recon,y_1)
        SSIM_ir = 1 - ssimloss(recon,ir_1)
        mse1=mse(y_1,recon)
        mse2 = mse(ir_1,recon)
        ssim_loss = torch.mean(w[:, 0] * SSIM_rgb + w[:, 1] * SSIM_ir)
        mse_loss = torch.mean(w[:, 0] * mse1 + w[:, 1] * mse2)
        # print(w[:, 0])
        # print(w[:, 1])
        # print(SSIM_rgb)
        # print(SSIM_ir)
        ##---------------------------融合loss-----------------------------------------

        # print(logits.shape)
        # print(labels.shape)
        # logits=probreweighting()(logits,labels)
        # logits=logits.to(args.device)
        # labels=labels.to(args.device)
        # loss=FocalLoss(gamma=1,alpha=1)(logits,labels)
        # loss = F.cross_entropy(logits, labels,weight=weights)
        pred=F.softmax(logits,dim=1)
        segloss=L.lovasz_softmax(pred,labels)
        loss = 20*mse_loss + ssim_loss+segloss
        # loss = L.lovasz_softmax(pred, labels)
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(logits, labels)
        # acc = 1
        loss_avg += float(loss)
        acc_avg  += float(acc)

        cur_t = time.time()
        if cur_t-t > 5:
            print('|- epo %s/%s. train iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
                % (epo, args.epoch_max, it+1, train_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss), float(acc)))
            t += 5
        predictions = logits.argmax(1)
        for gtcid in range(n_class):
            for pcid in range(n_class):
                gt_mask = labels == gtcid
                pred_mask = predictions == pcid
                intersection = gt_mask * pred_mask
                cf[gtcid, pcid] += int(intersection.sum())
    IoU = np.diag(cf) / (cf.sum(axis=1) + cf.sum(axis=0) - np.diag(cf))
    cls_recall=np.diag(cf)/(cf.sum(axis=1))

    recall=np.mean(cls_recall)
    print(recall)
    Miou = np.mean(IoU)
    content = '| epo:%s/%s lr:%.4f train_loss_avg:%.4f train_acc_avg:%.4f train_miou:%.4f\n ' \
              % (epo, args.epoch_max, lr_this_epo, loss_avg / train_loader.n_iter, acc_avg / train_loader.n_iter, Miou)
    # content = '| epo:%s/%s lr:%.4f train_loss_avg:%.4f train_acc_avg:%.4f \n ' \
    #           % (epo, args.epoch_max, lr_this_epo, loss_avg / train_loader.n_iter, acc_avg / train_loader.n_iter)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content)

    # content = '| epo:%s/%s lr:%.4f train_loss_avg:%.4f train_acc_avg:%.4f ' \
    #         % (epo, args.epoch_max, lr_this_epo, loss_avg/train_loader.n_iter, acc_avg/train_loader.n_iter)
    # print(content)
    # with open(log_file, 'a') as appender:
    #     appender.write(content)


def validation(epo, model_recon,model_seg, val_loader):
    ssimloss = SSIM(window_size=11)

    cf = np.zeros((n_class, n_class))
    loss_avg = 0.
    acc_avg  = 0.
    psnr_avg = 0.
    cc_avg = 0.
    ssim_avg = 0.
    start_t = time.time()
    model_recon.eval()
    model_seg.eval()
    mse = torch.nn.MSELoss(reduce=True, size_average=True)

    with torch.no_grad():
        # for it, (images, labels, names) in enumerate(val_loader):
        # for it, (images_yir, images,labels, names) in enumerate(val_loader):
        # for it, (images_yir,y_label,ir_label, names) in enumerate(val_loader):
        for it, (images_yir, labels, names) in enumerate(val_loader):
            # images = Variable(images)
            # labels = Variable(labels)
            # if args.gpu >= 0:
            images_yir = images_yir.to(args.device)
            # y_label = y_label.to(args.device)
            # ir_label = ir_label.to(args.device)
            # images = images.to(args.device)
            labels = labels.to(args.device)
            # rgb = images[:, :3]
            y = torch.cat((images_yir[:, :1], images_yir[:, :1], images_yir[:, :1]), dim=1)
            #
            ir = torch.cat((images_yir[:, 3:], images_yir[:, 3:], images_yir[:, 3:]), dim=1)
            y_1 = images_yir[:, :1]
            ir_1 = images_yir[:, 3:]
            cb = images_yir[:, 1].unsqueeze(1)
            cr = images_yir[:, 2].unsqueeze(1)
            reconinput = torch.cat((y_1, ir_1), dim=1)
            recon = model_recon(reconinput)
            seginput = torch.cat((recon, cb, cr), dim=1)
            logits = model_seg(seginput)
            # rgb = images_yir[:, :3]

            # ir = images_yir[:, 1:]

            # logits,reconstruct = model(images)
            # en = model.encoder(images_yir)
            # reconstruct = model.decoder(en)
            # reconstruct = model(images_yir)
            # logits = model(images_yir)
            # mse1 = mse(reconstruct,images_yir[:,:3])
            # mse2 = mse(reconstruct,ir)
            # loss = F.cross_entropy(logits, labels)
            # pred = F.softmax(logits, dim=1)
            # loss = L.lovasz_softmax(pred, labels)
            # acc = calculate_accuracy(logits, labels)
            # loss=mse1+mse2
            # SSIM_rgb = ssimloss(images_yir[:,:3], reconstruct)
            # SSIM_ir = ssimloss(ir, reconstruct)
            # ssim = (SSIM_rgb + SSIM_ir) / 2.0
            # b, c, h, w = reconstruct.size()
            # cc_rgb = pearsonr(reconstruct.view(b, c, -1), rgb.view(b, c, -1))
            # print(images_yir[:,:1].shape)
            # cc_ir = pearsonr(reconstruct.view(b, c, -1), ir.view(b, c, -1))
            # cc = (cc_rgb + cc_ir) / 2.0
            # print(cc.shape)
            # cc=torch.mean(cc)
            # images_yir = images_yir.cpu().numpy()
            # reconstruct = reconstruct.cpu().numpy()
            # rgb=rgb.cpu().numpy()
            # ir=ir.cpu().numpy()
            # y_label=y_label.cpu().numpy()
            # ir_label = ir_label.cpu().numpy()
            # psnrrgb = PSNR(rgb, reconstruct)
            # psnrir = PSNR(ir, reconstruct)
            # psnr = (psnrrgb + psnrir) / 2.0
            # acc = 1
            # psnr_avg += float(psnr)
            # cc_avg += float(cc)
            # ssim_avg += float(ssim)
            # loss_avg += float(loss)
            pred = F.softmax(logits, dim=1)
            loss = L.lovasz_softmax(pred, labels)
            acc = calculate_accuracy(logits, labels)
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
        IoU = np.diag(cf) / (cf.sum(axis=1) + cf.sum(axis=0) - np.diag(cf))
        val_cls_recall = np.diag(cf) / (cf.sum(axis=1))

        val_recall = np.mean(val_cls_recall)
        print(val_recall)
        Miou = np.mean(IoU)
        # Miou=0
        # val_recall=0
        content = '| val_loss_avg:%.4f val_acc_avg:%.4f val_miou:%.4f  recall:%.4f\n' \
                  % (loss_avg / val_loader.n_iter, acc_avg / val_loader.n_iter, Miou,val_recall)

        print(content)
        # content2 = '| val_cc_avg:%.4f val_ssim_avg:%.4f val_psnr_avg:%.4f\n' \
        #           % (cc_avg / val_loader.n_iter, ssim_avg / val_loader.n_iter, psnr_avg / val_loader.n_iter)
        # print(content2)
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
    rgbbackbone=backbonedensenet()
    irbackbone=backbonedensenet()


    model_seg = eval(args.model_name)(n_class=n_class)
    model_recon = DenseFuse_net(input_nc=2,output_nc=1)
    # if args.gpu >= 0:
    rgbbackbone=rgbbackbone.to(args.device)
    irbackbone = irbackbone.to(args.device)
    model_seg=model_seg.to(args.device)
    model_recon=model_recon.to(args.device)
    #-------加上lightnet----------------------------------------------------------------------------------
    # lightnet = LightNet()
    # # lightnet.train()
    # lightnet.to(args.device)
    # -------加上lightnet----------------------------------------------------------------------------------
    optimizer = torch.optim.SGD([
                {'params': model_recon.parameters()},
                {'params': model_seg.parameters()},
                {'params': rgbbackbone.parameters()},
                {'params': irbackbone.parameters()}
            ], lr=args.lr_start, momentum=0.9, weight_decay=0.0005)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, gamma=args.lr_decay, last_epoch=-1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
    lr_scheduler = get_scheduler(optimizer, args)

    if args.epoch_from > 1:
        print('| loading checkpoint file %s... ' % checkpoint_model_recon_file, end='')
        model_recon.load_state_dict(torch.load(checkpoint_model_recon_file))
        model_seg.load_state_dict(torch.load(checkpoint_model_seg_file))
        optimizer.load_state_dict(torch.load(checkpoint_optim_file))
        print('done!')
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    cropsize = [384, 288]
    # cropsize = [64,64]
    randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)
    train_dataset = MF_dataset_reconstruct(data_dir, cropsize=cropsize, mode='train', randomscale=randomscale)
    val_dataset = MF_dataset_reconstruct(data_dir, mode='test', randomscale=randomscale)
    # train_dataset = MF_dataset(data_dir, 'train', have_label=True, transform=augmentation_methods)
    # val_dataset  = MF_dataset(data_dir, 'test', have_label=True)

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
        batch_size  = args.batch_size_val,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    train_loader.n_iter = len(train_loader)
    val_loader.n_iter   = len(val_loader)
    miou=0
    rgb_gradloss=grad()
    ir_gradloss = grad()


    for epo in tqdm(range(args.epoch_from, args.epoch_max+1)):
        print('\n| epo #%s begin...' % epo)
        global best_miou, is_best
        c=3000

        train(epo, model_recon,model_seg,rgbbackbone,irbackbone, train_loader, optimizer,rgb_gradloss,ir_gradloss,c)
        lr_scheduler.step()
        if epo > 30 and epo % 2 == 0:
            miou = validation(epo, model_recon,model_seg, val_loader)
        elif epo <= 30 and epo % 5 == 0:
            miou = validation(epo, model_recon,model_seg, val_loader)
        # lr_scheduler.step()

        # save check point model
        is_best = miou > best_miou
        torch.save(model_recon.state_dict(), checkpoint_model_recon_file)
        torch.save(model_seg.state_dict(), checkpoint_model_seg_file)
        torch.save(optimizer.state_dict(), checkpoint_optim_file)
        print('done!')

        if is_best:
            best_miou = miou
            shutil.copy(checkpoint_model_recon_file, best_model_recon_file)
            shutil.copy(checkpoint_model_seg_file, best_model_seg_file)
            is_best = 0

    os.rename(checkpoint_model_recon_file, final_model_recon_file)
    os.rename(checkpoint_model_seg_file, final_model_seg_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train MFNet with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='mffenet_vgg')
    parser.add_argument('--backbone_name', '-b', type=str, default='backbonedensenet')
    parser.add_argument('--batch_size',  '-B',  type=int, default=8)
    parser.add_argument('--batch_size_val', '-Bv', type=int, default=4)
    parser.add_argument('--epoch_max' ,  '-E',  type=int, default=200)
    parser.add_argument('--epoch_from',  '-EF', type=int, default=1)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=0)
    parser.add_argument('--lr_power', default=0.9, type=float)
    parser.add_argument('--lr_policy', default='poly', type=str)
    parser.add_argument('--lr_start', '-ls', type=float, default=0.005)#0.005
    parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    args = parser.parse_args()

    model_dir = os.path.join(model_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_model_recon_file = os.path.join(model_dir, 'recontmp.pth')
    checkpoint_model_seg_file = os.path.join(model_dir, 'segtmp.pth')
    checkpoint_optim_file = os.path.join(model_dir, 'tmp.optim')
    final_model_recon_file      = os.path.join(model_dir, 'reconfinal.pth')
    final_model_seg_file = os.path.join(model_dir, 'segfinal.pth')
    log_file              = os.path.join(model_dir, 'log.txt')
    best_model_recon_file       = os.path.join(model_dir, 'reconbest.pth')
    best_model_seg_file = os.path.join(model_dir, 'segbest.pth')

    print('| training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('| from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('| model will be saved in: %s' % model_dir)

    main()
