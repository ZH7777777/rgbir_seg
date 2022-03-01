import os
import argparse
import time
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from util.MF_dataset import MF_dataset
from util.MF_dataset_day import MF_dataset_day
from util.MF_dataset_night import MF_dataset_night
from util.MF_dataset_val import MF_dataset_val
from util.util import calculate_accuracy
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise,RandomCrop3
from model import MFNet,discriminator, SegNet,RTFNet,afnet,SegNet,FuNNet,FuseSeg,afnetusetrans,fusesegusetrans,fusesegusetrans2,fusesegusetrans3,fusesegusetrans4
from model.relight import LightNet,L_TV,L_exp_z,SSIM
from model.discriminator import FCDiscriminator
from util.util import get_scheduler
import numpy as np
from tqdm import tqdm
from util.util import probreweighting,FocalLoss
from torch.utils import data
from tensorboardX import SummaryWriter
import torchvision
from util.util import StaticLoss

import torchvision.utils as vutils

def weightedMSE(D_out, D_label):
    return torch.mean((D_out - D_label).abs() ** 2)
# config
weights=torch.log(torch.FloatTensor([0.91865138 ,0.04526743, 0.01380293 ,0.00426743 ,0.00643076 ,0.00545441,
 0.00150976, 0.00175315, 0.00286275])).cuda()
weights = (torch.mean(weights) - weights) / torch.std(weights) * 0.05 + 1.0
n_class   = 9
# data_dir  = 'E:\google drive/ir_seg_dataset2/'
# model_dir = 'E:\MFNet-pytorch-master/weights/lightnet/'
data_dir  = '/content/drive/MyDrive/ir_seg_dataset2'
model_dir = '/content/drive/MyDrive/MFNet-pytorch-master/weights/lightnetlikedann/'
augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1,prob=1.0)



    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))
def adjust_learning_rate(args, optimizer, i_iter):
    lr = lr_poly(args.lr_start, i_iter, args.num_steps, args.lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
# lr_start  = 0.01
# lr_decay  = 0.95
day=0
night=1
def main():
    cf = np.zeros((n_class, n_class))
    random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    mioumax=0.35
    # lr_scheduler = get_scheduler(optimizer, args)
    # loss_exp_z = L_exp_z(32)
    # loss_TV = L_TV()
    # loss_SSIM = SSIM()

    model = eval(args.model_name)(n_class=n_class)
    # if args.gpu >= 0:
    # model.train()

    model=model.to(args.device)
    #-------加上lightnet----------------------------------------------------------------------------------
    lightnet = LightNet()
    # lightnet4ir = LightNet()
    # lightnet.train()
    lightnet.to(args.device)

    # lightnet4ir = LightNet()
    # lightnet4ir.train()
    # lightnet4ir.to(args.device)
    #判别器-----------------------------------------------------------------------------------------
    model_D1 = FCDiscriminator(num_classes=n_class)
    # model_D1.train()
    model_D1.to(args.device)
    # ---------------------------------------加载模型------------------------------------------------------

    # -------加上lightnet----------------------------------------------------------------------------------
    optimizer = torch.optim.SGD(list(model.parameters())+list(lightnet.parameters()), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)#lightnet参数加入更新
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, gamma=args.lr_decay, last_epoch=-1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
    optimizer.zero_grad()
    optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    # lr_scheduler4g = get_scheduler(optimizer, args)
    # lr_scheduler4d = get_scheduler(optimizer_D1, args)
    #-------------------------加载模型---------------------------------------------------------------------------------
    if args.epoch_from > 1:
        print('| loading checkpoint file %s... ' % checkpoint_model_file, end='')
        model.load_state_dict(torch.load(checkpoint_model_file))
        lightnet.load_state_dict(torch.load(checkpoint_lightnet_file))
        # lightnet4ir.load_state_dict(torch.load(checkpoint_lightnet4ir_file))
        optimizer.load_state_dict(torch.load(checkpoint_optim_file))
        print('done!')
    #--------------------------------------------------------------------------------------------------------------

    loss_exp_z = L_exp_z(32)
    loss_TV = L_TV()
    loss_SSIM = SSIM()
    static_loss = StaticLoss(num_classes=5, weight=weights[:5])


    # if args.epoch_from > 1:
    #     print('| loading checkpoint file %s... ' % checkpoint_model_file, end='')
    #     model.load_state_dict(torch.load(checkpoint_model_file))
    #     lightnet.load_state_dict(torch.load(checkpoint_lightnet_file))
    #     lightnet4ir.load_state_dict(torch.load(checkpoint_lightnet4ir_file))
    #     optimizer.load_state_dict(torch.load(checkpoint_optim_file))
    #     print('done!')

    train_dataset_day = MF_dataset_day(data_dir, 'train_day', max_iters=args.num_steps * args.iter_size * args.batch_size, have_label=True, transform=augmentation_methods)
    train_dataset_night = MF_dataset_night(data_dir, 'train_night', max_iters=args.num_steps * args.iter_size * args.batch_size, have_label=True, transform=augmentation_methods)
    val_dataset  = MF_dataset_val(data_dir, 'val', max_iters=args.num_steps * args.iter_size * args.batch_size,have_label=True)

    train_loader_day  = data.DataLoader(
        dataset     = train_dataset_day,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True
    )
    trainloader_day_iter = enumerate(train_loader_day)
    train_loader_night = data.DataLoader(
        dataset=train_dataset_night,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    trainloader_night_iter = enumerate(train_loader_night)
    val_loader  = data.DataLoader(
        dataset     = val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    # val_loader_iter=enumerate(val_loader)
    val_loader.n_iter = len(val_loader)
    val_loader_iter = enumerate(val_loader)
    # if args.epoch_from > 1:
    #     print('| loading checkpoint file %s... ' % checkpoint_model_file, end='')
    #     model.load_state_dict(torch.load(checkpoint_model_file))
    #     lightnet.load_state_dict(torch.load(checkpoint_lightnet_file))
    #     lightnet4ir.load_state_dict(torch.load(checkpoint_lightnet4ir_file))
    #     optimizer.load_state_dict(torch.load(checkpoint_optim_file))
    #     print('done!')
   #train  val--------------------------------------------------------------------------------------
    # rgb_night_mean_light = 0.1003
    # rgb_day_mean_light = 0.3612
    # ir_night_mean_light = 0.2725
    # ir_day_mean_light = 0.5075
    rgb_night_mean_light = 0.2003
    rgb_day_mean_light = 0.3612
    ir_night_mean_light = 0.2725
    ir_day_mean_light = 0.4075
    rgb_mean_light = 0.2222
    ir_mean_light = 0.2725
    for i_iter in range(args.num_steps):
        # print('\n| epo #%s begin...' % epo)
        loss_seg_value = 0
        loss_segnight_value = 0
        loss_adv_target_value = 0
        loss_adv_target_day=0
        loss_adv_target_night = 0

        loss_D_value_day = 0
        loss_D_value_night = 0
        model.train()
        lightnet.train()
        # lightnet4ir.train()
        model_D1.train()



        optimizer.zero_grad()
        optimizer_D1.zero_grad()
        adjust_learning_rate(args, optimizer, i_iter)
        adjust_learning_rate_D(args, optimizer_D1, i_iter)
        #train G

        for sub_i in range(args.iter_size):
            for param in model_D1.parameters():
                param.requires_grad = False
            #day------------------------------
            _, batch = trainloader_day_iter.__next__()
            _, batch2 = trainloader_night_iter.__next__()
            image_d, label_d, name_d = batch
            image_n, label_n, name_n = batch2
            print(name_d)
            print(name_n)
            label_d=label_d.to(args.device)
            label_n = label_n.to(args.device)
            image_d = image_d.to(args.device)
            image_n = image_n.to(args.device)
            rgb_d = image_d[:, :3]
            ir_d = image_d[:, 3:]
            rgb_n = image_n[:, :3]
            ir_n = image_n[:, 3:]

            mean_light = rgb_n.mean()
            ir_d = torch.cat((ir_d, ir_d, ir_d), dim=1)  # b*3*h*w
            ir_n= torch.cat((ir_n, ir_n, ir_n), dim=1)

            r_d= lightnet(rgb_d)
            # r_ir = lightnet4ir(ir_d)
            enhanced_images_rgb_d = rgb_d + r_d
            # enhanced_images_ir = ir + r_ir
            ##--------------------------------tensorboard可视化-------------
            # ir4view=torch.cat((ir,ir,ir),dim=1)
            
            
            grid1=torch.cat((rgb_d,r_d,enhanced_images_rgb_d,ir_d),dim=0)
            # grid=r
            
            
            # grid = torchvision.utils.make_grid(grid,nrow=2, normalize=True)
            # print(grid.shape)
            # writer.add_image('rgb',grid,global_step=i_iter)
            # writer.close()
            #----------------------------------------------------------------------------
            # loss_enhance_rgb = 10 * loss_TV(r) + torch.mean(loss_SSIM(enhanced_images_rgb, rgb)) \
            #                    + torch.mean(loss_exp_z(enhanced_images_rgb, 0.5))
            loss_enhance_rgb = 10 * loss_TV(r_d)+torch.mean(loss_SSIM(enhanced_images_rgb_d, rgb_d))+torch.mean(loss_exp_z(enhanced_images_rgb_d, mean_light))
            # loss_enhance_rgb.backward()
            # r2 = lightnet(ir)
            # enhanced_images_ir = ir + r2
            # loss_enhance_ir = 10 * loss_TV(r_ir) + torch.mean(loss_SSIM(enhanced_images_ir, ir)) \
            #                   + torch.mean(loss_exp_z(enhanced_images_ir, 0.5))
            # loss_enhance_ir = 0.01 * loss_TV(ir)+torch.mean(loss_SSIM(enhanced_images_ir, ir))+torch.mean(loss_exp_z(enhanced_images_ir, ir_mean_light))
            # loss_enhance_ir.backward()

            ir_d = ir_d[:, :1, :, :]

            loss_enhance =  loss_enhance_rgb
            print(loss_enhance)
            # loss.backward()
            

            images_d = torch.cat((enhanced_images_rgb_d, ir_d), dim=1)
            #
            #
            #
            # # --------------------------------lightnet-----------------------------
            logits_day = model(images_d)
            D_out_d = model_D1(F.softmax(logits_day, dim=1))
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(night).to(args.device)
            loss_adv_target_d = weightedMSE(D_out_d, D_label_d)
            # print(loss_adv_target_d)
            seg_loss = F.cross_entropy(logits_day, label_d, weight=weights)
            loss = seg_loss+0.01*loss_enhance+0.01*loss_adv_target_d
            loss=loss / args.iter_size
            # seg_loss=0
            # loss_seg_value += seg_loss.item() / args.iter_size
            loss_seg_value += 0
            # loss_adv_target_day += loss_adv_target_d.item() / args.iter_size

            # print(1)

            loss.backward()
            #day--------------------------

            #night----------------------

            r_n= lightnet(rgb_n)
            # r_ir = lightnet4ir(ir)
            enhanced_images_rgb_n = rgb_n + r_n
            # enhanced_images_ir = ir + r_ir
            ##--------------------------------tensorboard可视化-------------
            # ir4view=torch.cat((ir,ir,ir),dim=1)
            grid2=torch.cat((rgb_n,r_n,enhanced_images_rgb_n,ir_n),dim=0)
            # grid=r
            grid=torch.cat((grid1,grid2),dim=0)
            
            
            grid = torchvision.utils.make_grid(grid,nrow=2, normalize=True)
            # print(grid.shape)
            writer.add_image('rgb',grid,global_step=i_iter)
            # writer.close()
            #-----------------------------------------------------------------------------------------------
            # loss_enhance_rgb = 10 * loss_TV(r) + torch.mean(loss_SSIM(enhanced_images_rgb, rgb)) \
            #                    + torch.mean(loss_exp_z(enhanced_images_rgb, 0.5))
            loss_enhance_rgb_n = 10 * loss_TV(r_n)+torch.mean(loss_SSIM(enhanced_images_rgb_n, rgb_n))+torch.mean(loss_exp_z(enhanced_images_rgb_n,mean_light))
            # loss_enhance_rgb.backward()
            # r2 = lightnet(ir)
            # enhanced_images_ir = ir + r2
            # loss_enhance_ir = 10 * loss_TV(r_ir) + torch.mean(loss_SSIM(enhanced_images_ir, ir)) \
            #                   + torch.mean(loss_exp_z(enhanced_images_ir, 0.5))
            # loss_enhance_ir = 0.01 * loss_TV(ir)+torch.mean(loss_SSIM(enhanced_images_ir, ir))+torch.mean(loss_exp_z(enhanced_images_ir, ir_mean_light2))
            # loss_enhance_ir.backward()

            ir_n = ir_n[:, :1, :, :]

            loss_enhance = loss_enhance_rgb_n
            # loss.backward()
            print(loss_enhance)

            images_n = torch.cat((enhanced_images_rgb_n, ir_n), dim=1)
            #
            #
            #
            # # --------------------------------lightnet-----------------------------
            logits_night = model(images_n)
            #白天的分割结果作为晚上的label
            psudo_prob = torch.zeros_like(logits_day)
            threshold = torch.ones_like(logits_day[:, :5, :, :]) * 0.2
            threshold[logits_day[:, :5, :, :] > 0.4] = 0.8
            psudo_prob[:, :5, :, :] = threshold * logits_day[:, :5, :, :].detach() + (
                        1 - threshold) * logits_night[:, :5, :, :].detach()
            psudo_prob[:, 5:, :, :] = logits_night[:, 5:, :, :].detach()

            weights_prob = weights.expand(psudo_prob.size()[0], psudo_prob.size()[3], psudo_prob.size()[2], 9)
            weights_prob = weights_prob.transpose(1, 3)
            psudo_prob = psudo_prob * weights_prob
            psudo_gt = torch.argmax(psudo_prob.detach(), dim=1)
            psudo_gt[psudo_gt >= 5] = 255
            #------------------------------------
            D_out_d = model_D1(F.softmax(logits_night, dim=1))
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(day).to(args.device)
            loss_adv_target_d = weightedMSE(D_out_d, D_label_d)
            # seg_loss= F.cross_entropy(logits_night, labels,weight=weights)
            loss_pseudo = static_loss(logits_night[:, :5, :, :], psudo_gt)
            loss =loss_pseudo+0.01*loss_enhance+0.01*loss_adv_target_d
            loss = loss / args.iter_size
            loss_segnight_value += seg_loss.item() / args.iter_size
            # loss_adv_target_night += loss_adv_target_d.item() / args.iter_size
            # print(2)
            loss.backward()
            #-----------------------------------------------------------------------------

            # train D
            for param in model_D1.parameters():
                param.requires_grad = True
            logits_night=logits_night.detach()
            D_out1 = model_D1(F.softmax(logits_night, dim=1))
            D_label1 = torch.FloatTensor(D_out1.data.size()).fill_(night).to(args.device)
            loss_D1 = weightedMSE(D_out1, D_label1)
            loss_D1 = loss_D1 / args.iter_size/2
            loss_D1.backward()
            loss_D_value_night += loss_D1.item()
            
            logits_day = logits_day.detach()
            D_out1 = model_D1(F.softmax(logits_day, dim=1))
            D_label1 = torch.FloatTensor(D_out1.data.size()).fill_(day).to(args.device)
            loss_D1 = weightedMSE(D_out1, D_label1)
            loss_D1 = loss_D1 / args.iter_size/2
            loss_D1.backward()
            loss_D_value_day += loss_D1.item()
        optimizer.step()
        optimizer_D1.step()
        # print(
        #     'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_day = {3:.3f},loss_adv_night= {4:.3f}, loss_D1 = {5:.3f},loss_D1night = {6:.3f}'.format(
        #         i_iter, args.num_steps, loss_seg_value,
        #         loss_adv_target_day,loss_adv_target_night ,loss_D_value_day,loss_D_value_night))
        print(
            'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value,
                ))

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('| saving check point model file... ', end='')
            torch.save(model.state_dict(), checkpoint_model_file)
            torch.save(optimizer.state_dict(), checkpoint_optim_file)
            torch.save(lightnet.state_dict(), checkpoint_lightnet_file)
            # torch.save(lightnet4ir.state_dict(), checkpoint_lightnet4ir_file)
            # torch.save(model_D1.state_dict(), checkpoint_model_D1_file)
            # torch.save(optimizer_D1.state_dict(), checkpoint_optimD1_file)
            print('done!')
        # lr_scheduler.step()
        # if epo>40:
        #
        #     miou=validation(epo, model,lightnet,lightnet4ir, val_loader)
        #     if miou>=mioumax :
        #         mioumax=miou
        #         torch.save(model.state_dict(), best_model_file)
        #         torch.save(lightnet.state_dict(), best_lightnet_file)
        #         torch.save(lightnet4ir.state_dict(), best_lightnet_file)
        # # lr_scheduler.step()
        #
        # # save check point model
        # print('| saving check point model file... ', end='')
        # torch.save(model.state_dict(), checkpoint_model_file)
        # torch.save(optimizer.state_dict(), checkpoint_optim_file)
        # torch.save(lightnet.state_dict(), checkpoint_lightnet_file)
        # # torch.save(lightnet4ir.state_dict(), checkpoint_lightnet4ir_file)
        # print('done!')
        # lr_scheduler.step()
        #--------------------------------------------------------val---------------------------------
        if i_iter>16000 and i_iter%args.val_per_iter==0:
            for i_iter2 in range(args.val_steps):
                loss_avg = 0.
                acc_avg = 0.
                start_t = time.time()

                model.eval()
                lightnet.eval()
                # lightnet4ir.eval()
                loss_exp_z = L_exp_z(32)
                loss_TV = L_TV()
                loss_SSIM = SSIM()
                with torch.no_grad():
                    for sub_i in range(args.iter_size):

                        # day------------------------------
                        _, batch = val_loader_iter.__next__()
                        image, label, name = batch
                        images = image.to(args.device)
                        labels = label.to(args.device)

                        # # --------------------------------lightnet-----------------------------
                        rgb = images[:, :3]
                        ir = images[:, 3:]
                        ir = torch.cat((ir, ir, ir), dim=1)  # b*3*h*w

                        r = lightnet(rgb)
                        # r_ir = lightnet4ir(ir)
                        enhanced_images_rgb = rgb + r
                        # enhanced_images_ir = ir + r_ir



                        enhanced_images_ir = ir[:, :1, :, :]
                        # grid3=torch.cat((rgb,r,enhanced_images_rgb,ir,r_ir,enhanced_images_ir),dim=0)
                        
                        # grid=torch.cat((grid1,grid2,grid3),dim=0)
                        
                        
                        # grid = torchvision.utils.make_grid(grid,nrow=2, normalize=True)
                        # print(grid.shape)
                        # writer.add_image('rgb',grid,global_step=i_iter)

                        loss_enhance = enhanced_images_rgb
                        # loss.backward()

                        images = torch.cat((enhanced_images_rgb, enhanced_images_ir), dim=1)
                        #
                        # # --------------------------------lightnet-----------------------------
                        logits = model(images)

                        seg_loss = F.cross_entropy(logits, labels, weight=weights)
                        acc = calculate_accuracy(logits, labels)
                        loss_avg += float(seg_loss)
                        acc_avg += float(acc)

                        cur_t = time.time()
                        print('|- iter %s val iter %s  loss: %.4f, acc: %.4f' \
                              % (i_iter2, args.val_steps,
                                 float(seg_loss), float(acc)))
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
            if Miou>mioumax:
                mioumax = Miou
                torch.save(model.state_dict(), best_model_file)
                torch.save(lightnet.state_dict(), best_lightnet_file)
                # torch.save(lightnet4ir.state_dict(), best_lightnet4ir_file)
                torch.save(model_D1.state_dict(), best_model_D1_file)
    writer.close()
    os.rename(checkpoint_model_file, final_model_file)
    os.rename(checkpoint_lightnet_file, final_lightnet_file)
    # os.rename(checkpoint_lightnet4ir_file, final_lightnet4ir_file)
    # os.rename(checkpoint_model_D1_file, final_model_D1_file)







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
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_steps', default='20000',type=int,
                        help='train')
    parser.add_argument('--val_steps', default='196',type=int,
                        help='val')
    parser.add_argument('--iter_size', default='1',type=int,
                        help='train')
    parser.add_argument('--val_per_iter', default='200',type=int,
                        help='val')
    parser.add_argument('--save_pred_every', default='4000', type=int,
                        help='save')


    parser.add_argument("--learning-rate-D", type=float, default=1e-4,
                        help="Base learning rate for discriminator.")
    args = parser.parse_args()

    model_dir = os.path.join(model_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir='tb/trainlightnet/2')

    checkpoint_model_file = os.path.join(model_dir, 'tmp.pth')
    checkpoint_optim_file = os.path.join(model_dir, 'tmp.optim')
    checkpoint_optimD1_file = os.path.join(model_dir, 'D1tmp.optim')
    final_model_file      = os.path.join(model_dir, 'final.pth')
    log_file              = os.path.join(model_dir, 'log.txt')
    best_model_file       = os.path.join(model_dir, 'best.pth')
    checkpoint_lightnet_file = os.path.join(model_dir, 'lightnettmp.pth')
    final_lightnet_file = os.path.join(model_dir, 'lightnetfinal.pth')
    best_lightnet_file = os.path.join(model_dir, 'lightnetbest.pth')
    checkpoint_lightnet4ir_file = os.path.join(model_dir, 'lightnet4irtmp.pth')
    final_lightnet4ir_file = os.path.join(model_dir, 'lightnet4irfinal.pth')
    best_lightnet4ir_file = os.path.join(model_dir, 'lightnet4irbest.pth')
    checkpoint_model_D1_file = os.path.join(model_dir, 'model_D1tmp.pth')
    final_model_D1_file = os.path.join(model_dir, 'model_D1final.pth')
    best_model_D1_file = os.path.join(model_dir, 'model_D1best.pth')

    print('| training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('| from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('| model will be saved in: %s' % model_dir)

    main()