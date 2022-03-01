import os
import argparse
import time
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from util.MF_dataset import MF_dataset
from util.util import calculate_accuracy
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise,RandomCrop3
from model import MFNet, SegNet,RTFNet,afnet,SegNet,FuNNet,FuseSeg,afnetusetrans,fusesegusetrans,fusesegusetrans2,fusesegusetrans3,fusesegusetrans4
from model.relight import LightNet,L_TV,L_exp_z,SSIM
from util.util import get_scheduler
import numpy as np
from tqdm import tqdm
from util.util import probreweighting,FocalLoss

# config
weights=torch.log(torch.FloatTensor([0.91865138 ,0.04526743, 0.01380293 ,0.00426743 ,0.00643076 ,0.00545441,
 0.00150976, 0.00175315, 0.00286275])).cuda()
weights = (torch.mean(weights) - weights) / torch.std(weights) * 0.05 + 1.0
n_class   = 9
data_dir  = 'E:\google drive/ir_seg_dataset2/'
model_dir = 'E:\MFNet-pytorch-master/weights/lightnet/'
# data_dir  = '/content/drive/MyDrive/ir_seg_dataset2'
# model_dir = '/content/drive/MyDrive/MFNet-pytorch-master/weights/lightnet/'
augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1,prob=1.0)



    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]
# lr_start  = 0.01
# lr_decay  = 0.95


def train(epo, model,lightnet,lightnet4ir, train_loader, optimizer):

    # lr_this_epo = lr_start * lr_decay**(epo-1)
    for param_group in optimizer.param_groups:

        lr_this_epo=param_group['lr']

    lightnet_loss_avg=0
    lightnet4ir_loss_avg = 0
    loss_avg = 0.
    acc_avg  = 0.
    start_t = t = time.time()
    cf = np.zeros((n_class, n_class))
    model.train()
    lightnet.train()
    lightnet4ir.train()

    for it, (images, labels, names) in enumerate(train_loader):
        # images = Variable(images).cuda()
        # labels = Variable(labels).cuda()
        # if args.gpu >= 0:
        #name 判断 是day 还是night-------------

        name=names
        name1=names[0]
        name2=names[1]
        a=name1[5]
        #--------------------------------------------------
        loss_exp_z = L_exp_z(32)
        loss_TV = L_TV()
        loss_SSIM = SSIM()
        images = images.to(args.device)
        labels = labels.to(args.device)
        # # #--------------------------------lightnet-----------------------------
        # # rgb = images[:, :3]
        # # ir = images[:, 3:]
        # # mean_light = rgb.mean()
        # # r = lightnet(rgb)
        # # enhanced_images_rgb = rgb + r
        # # loss_enhance = 10 * loss_TV(r) + torch.mean(loss_SSIM(enhanced_images_rgb, rgb)) \
        # #                + torch.mean(loss_exp_z(enhanced_images_rgb, mean_light))
        # #
        # # # --------------------------------lightnet-----------------------------
        #
        optimizer.zero_grad()
        # # --------------------------------lightnet-----------------------------
        rgb = images[:, :3]
        ir = images[:, 3:]
        ir=torch.cat((ir,ir,ir),dim=1)#b*3*h*w
        rgb_mean_light = 0.1003
        ir_mean_light = 0.2725
        r = lightnet(rgb)
        r_ir = lightnet4ir(ir)
        enhanced_images_rgb = rgb + r
        enhanced_images_ir = ir + r_ir
        loss_enhance_rgb = 10 * loss_TV(r) + torch.mean(loss_SSIM(enhanced_images_rgb, rgb)) \
                       + torch.mean(loss_exp_z(enhanced_images_rgb, rgb_mean_light))
        # loss_enhance_rgb.backward()
        # r2 = lightnet(ir)
        # enhanced_images_ir = ir + r2
        loss_enhance_ir = 10 * loss_TV(r_ir) + torch.mean(loss_SSIM(enhanced_images_ir, ir)) \
                       + torch.mean(loss_exp_z(enhanced_images_ir, ir_mean_light))
        # loss_enhance_ir.backward()

        enhanced_images_ir=enhanced_images_ir[:,:1,:,:]

        # loss = 0.1 * enhanced_images_ir + 0.1 * enhanced_images_rgb
        # loss.backward()

        images=torch.cat((enhanced_images_rgb,enhanced_images_ir),dim=1)
        #
        #
        #
        # # --------------------------------lightnet-----------------------------
        logits = model(images)
        # logits=probreweighting()(logits,labels)
        # logits=logits.to(args.device)
        # labels=labels.to(args.device)
        # loss=FocalLoss(gamma=1,alpha=1)(logits,labels)
        loss = F.cross_entropy(logits, labels,weight=weights)+0.1*loss_enhance_rgb+0.1*loss_enhance_ir
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(logits, labels)
        loss_avg += float(loss)
        lightnet_loss_avg+=float(loss_enhance_rgb)
        lightnet4ir_loss_avg += float(loss_enhance_ir)

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
    Miou = np.mean(IoU)
    content = '| epo:%s/%s lr:%.4f train_loss_avg:%.4f lightnet_loss_avg:%.4f lightnet4ir_loss_avg:%.4f train_acc_avg:%.4f train_miou:%.4f\n' \
              % (epo, args.epoch_max, lr_this_epo, loss_avg / train_loader.n_iter, lightnet_loss_avg / train_loader.n_iter,lightnet4ir_loss_avg / train_loader.n_iter,acc_avg / train_loader.n_iter, Miou)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content)

    # content = '| epo:%s/%s lr:%.4f train_loss_avg:%.4f train_acc_avg:%.4f ' \
    #         % (epo, args.epoch_max, lr_this_epo, loss_avg/train_loader.n_iter, acc_avg/train_loader.n_iter)
    # print(content)
    # with open(log_file, 'a') as appender:
    #     appender.write(content)


def validation(epo, model,lightnet,lightnet4ir, val_loader):

    cf = np.zeros((n_class, n_class))
    loss_avg = 0.
    acc_avg  = 0.
    start_t = time.time()
    model.eval()
    lightnet.eval()
    lightnet4ir.eval()
    loss_exp_z = L_exp_z(32)
    loss_TV = L_TV()
    loss_SSIM = SSIM()

    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            # images = Variable(images)
            # labels = Variable(labels)
            # if args.gpu >= 0:
            images = images.to(args.device)
            labels = labels.to(args.device)
            # # --------------------------------lightnet-----------------------------


            # --------------------------------lightnet-----------------------------
            rgb = images[:, :3]
            ir = images[:, 3:]
            ir = torch.cat((ir, ir, ir), dim=1)  # b*3*h*w
            rgb_mean_light = rgb.mean()
            ir_mean_light = ir.mean()
            r = lightnet(rgb)
            r_ir = lightnet4ir(ir)
            enhanced_images_rgb = rgb + r
            enhanced_images_ir = ir + r_ir
            loss_enhance_rgb = 10 * loss_TV(r) + torch.mean(loss_SSIM(enhanced_images_rgb, rgb)) \
                               + torch.mean(loss_exp_z(enhanced_images_rgb, rgb_mean_light))
            # loss_enhance_rgb.backward()
            # r2 = lightnet(ir)
            # enhanced_images_ir = ir + r2
            loss_enhance_ir = 10 * loss_TV(r_ir) + torch.mean(loss_SSIM(enhanced_images_ir, ir)) \
                              + torch.mean(loss_exp_z(enhanced_images_ir, ir_mean_light))
            # loss_enhance_ir.backward()

            enhanced_images_ir = enhanced_images_ir[:, :1, :, :]

            # loss = 0.1 * enhanced_images_ir + 0.1 * enhanced_images_rgb
            # loss.backward()

            images = torch.cat((enhanced_images_rgb, enhanced_images_ir), dim=1)

            # --------------------------------lightnet-----------------------------

            logits = model(images)
            loss = F.cross_entropy(logits, labels)+0.5*loss_enhance_rgb+0.5*loss_enhance_ir
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
    # loss_exp_z = L_exp_z(32)
    # loss_TV = L_TV()
    # loss_SSIM = SSIM()

    model = eval(args.model_name)(n_class=n_class)
    # if args.gpu >= 0:
    model=model.to(args.device)
    #-------加上lightnet----------------------------------------------------------------------------------
    lightnet = LightNet()
    # lightnet4ir = LightNet()
    lightnet.to(args.device)

    lightnet4ir = LightNet()
    lightnet4ir.to(args.device)
    # -------加上lightnet----------------------------------------------------------------------------------
    optimizer = torch.optim.SGD(list(model.parameters())+list(lightnet4ir.parameters())+list(lightnet.parameters()), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)#lightnet参数加入更新
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, gamma=args.lr_decay, last_epoch=-1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
    lr_scheduler = get_scheduler(optimizer, args)

    if args.epoch_from > 1:
        print('| loading checkpoint file %s... ' % checkpoint_model_file, end='')
        model.load_state_dict(torch.load(checkpoint_model_file))
        lightnet.load_state_dict(torch.load(checkpoint_lightnet_file))
        lightnet4ir.load_state_dict(torch.load(checkpoint_lightnet4ir_file))
        optimizer.load_state_dict(torch.load(checkpoint_optim_file))
        print('done!')

    train_dataset = MF_dataset(data_dir, 'train', have_label=True, transform=augmentation_methods)
    val_dataset  = MF_dataset(data_dir, 'val', have_label=True)

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

        train(epo, model,lightnet,lightnet4ir, train_loader, optimizer)
        # lr_scheduler.step()
        if epo>40:

            miou=validation(epo, model,lightnet,lightnet4ir, val_loader)
            if miou>=mioumax :
                mioumax=miou
                torch.save(model.state_dict(), best_model_file)
                torch.save(lightnet.state_dict(), best_lightnet_file)
                torch.save(lightnet4ir.state_dict(), best_lightnet4ir_file)
        # lr_scheduler.step()

        # save check point model
        print('| saving check point model file... ', end='')
        torch.save(model.state_dict(), checkpoint_model_file)
        torch.save(optimizer.state_dict(), checkpoint_optim_file)
        torch.save(lightnet.state_dict(), checkpoint_lightnet_file)
        torch.save(lightnet4ir.state_dict(), checkpoint_lightnet4ir_file)
        print('done!')
        lr_scheduler.step()

    os.rename(checkpoint_model_file, final_model_file)
    os.rename(checkpoint_lightnet_file, final_lightnet_file)
    os.rename(checkpoint_lightnet4ir_file, final_lightnet4ir_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train MFNet with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='FuseSeg')
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
    checkpoint_lightnet_file = os.path.join(model_dir, 'lightnettmp.pth')
    final_lightnet_file = os.path.join(model_dir, 'lightnetfinal.pth')
    best_lightnet_file = os.path.join(model_dir, 'lightnetbest.pth')
    checkpoint_lightnet4ir_file = os.path.join(model_dir, 'lightnet4irtmp.pth')
    final_lightnet4ir_file = os.path.join(model_dir, 'lightnet4irfinal.pth')
    best_lightnet4ir_file = os.path.join(model_dir, 'lightnet4irbest.pth')

    print('| training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('| from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('| model will be saved in: %s' % model_dir)

    main()