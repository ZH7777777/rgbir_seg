# coding:utf-8
import os
import argparse
import time
import numpy as np
from thop import profile
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from util.MF_dataset import MF_dataset
from util.MF_dataset_new import MF_dataset_new
from util.MF_dataset_reconstruct import MF_dataset_reconstruct
from util.util import calculate_accuracy, calculate_result

from model import backbonedensenetencoder,recon_decoder,seg_decoder,DenseFuse_net, FuseSeg_cat, mffenet_vgg, pool_mffenet, ccaffmnet, mffenetusetrans, mffenet, \
    mffenetusetrans_c, FuseSeguseboundary, mffenetusecswintrans, fusesegusetrans_new, fusesegusetrans5, \
    fusesegusetrans2, MFNet, RTFNet, afnet, FuseSeg, FuNNet, afnetusetrans, fusesegusetrans, fusesegusetrans3, \
    fusesegusetrans4
from train_reconandsegendecoder import n_class, data_dir, model_dir
from util.util import label2rgb
# from util.util import visualize,visualize2,visualize_fusesegboundary
from util.utilvisualize import visualize, visualize_reconstrucion, tensor2img, visualize_gray
import cv2
import math
from ssim import SSIM
from audtorch.metrics.functional import pearsonr


def PSNR(target, ref):
    target = np.array(target, dtype=np.float32)
    ref = np.array(ref, dtype=np.float32)

    if target.shape != ref.shape:
        raise ValueError('输入图像的大小应该一致！')

    diff = ref - target
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))
    psnr = 20 * math.log10(1.0 / rmse)

    return psnr


def main():
    ssimloss = SSIM(window_size=11)

    cf = np.zeros((n_class, n_class))

    # model = eval(args.model_name)(n_class=n_class)
    encoder = backbonedensenetencoder(dim=2)
    rec_decoder = recon_decoder()
    se_decoder = seg_decoder()
    encoder = encoder.to(args.gpu)
    rec_decoder = rec_decoder.to(args.gpu)
    se_decoder = se_decoder.to(args.gpu)
    # model = DenseFuse_net(input_nc=4, output_nc=3)
    # if args.gpu >= 0: model.cuda(args.gpu)
    print('| loading model file %s... ' % final_model_encoder_file, end='')
    # model.load_state_dict(torch.load(final_model_file, map_location={'cuda:0':'cuda:1'}))
    encoder.load_state_dict(torch.load(final_model_encoder_file))
    rec_decoder.load_state_dict(torch.load(final_model_rec_decoder_file))
    se_decoder.load_state_dict(torch.load(final_model_se_decoder_file))
    # print(model)
    print('done!')

    randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)

    # test_dataset = MF_dataset_new(data_dir, mode='test', randomscale=randomscale)
    # test_dataset  = MF_dataset(data_dir, 'test', have_label=True)
    test_dataset = MF_dataset_reconstruct(data_dir, mode='test', randomscale=randomscale)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    test_loader.n_iter = len(test_loader)

    loss_avg = 0.
    acc_avg = 0.
    psnr_avg = 0.
    cc_avg = 0.
    ssim_avg = 0.
    encoder.eval()
    rec_decoder.eval()
    se_decoder.eval()
    # #parameters-------------------------------------------
    # input=torch.randn(1,4,480,640).cuda()
    # macs,params=profile(model,inputs=(input,))
    # print("macs",macs)
    # print("params",params)
    # #-------------------------------------------------------
    # #speed test--------------------------------------------
    # time_spent=[]
    # iteration=100
    # for _ in range(iteration):
    #     t_start=time.time()
    #     with torch.no_grad():
    #         model(input)
    #     torch.cuda.synchronize()
    #     time_spent.append(time.time()-t_start)
    # elapsed_time=np.mean(time_spent)
    # print('Elapsed time:[%.2f s/%d iter]'%(elapsed_time,iteration))
    # print('speed time:%.4f s/iter    fps:%.2f'%(elapsed_time,1/elapsed_time))
    # #-----------------------------------------------------------------------------
    with torch.no_grad():
        # for it, (images_yir,images, labels, names) in enumerate(test_loader):
        # for it, (images_yir,y_label,ir_label, names) in enumerate(test_loader):
        for it, (images_yir, labels, names) in enumerate(test_loader):
            # images = Variable(images)
            # labels = Variable(labels)
            # if args.gpu >= 0:
            images_yir = images_yir.to(args.gpu)
            # y_label = y_label.to(args.device)
            # ir_label = ir_label.to(args.device)
            # images = images.to(args.device)
            labels = labels.to(args.gpu)
            # rgb = images[:, :3]
            y = torch.cat((images_yir[:, :1], images_yir[:, :1], images_yir[:, :1]), dim=1)
            #
            ir = torch.cat((images_yir[:, 3:], images_yir[:, 3:], images_yir[:, 3:]), dim=1)
            y_1 = images_yir[:, :1]
            ir_1 = images_yir[:, 3:]
            cb = images_yir[:, 1].unsqueeze(1)
            cr = images_yir[:, 2].unsqueeze(1)
            reconinput = torch.cat((y_1, ir_1), dim=1)
            rgb1, rgb2, rgb3, rgb4 = encoder(reconinput)

            # seginput = torch.cat((recon, cb, cr), dim=1)
            reconstruct=rec_decoder(rgb1, rgb2, rgb3, rgb4)
            logits = se_decoder(rgb1, rgb2, rgb3, rgb4)

            # reconstruct = model(images_yir)
            # --------------------reconstruct eval------------

            SSIM_rgb = ssimloss(y_1, reconstruct)
            SSIM_ir = ssimloss(ir_1, reconstruct)
            ssim = (SSIM_rgb + SSIM_ir) / 2.0
            b, c, h, w = reconstruct.size()
            cc_rgb = pearsonr(reconstruct.view(b, c, -1), y_1.view(b, c, -1))
            # print(images_yir[:,:1].shape)
            cc_ir = pearsonr(reconstruct.view(b, c, -1), ir_1.view(b, c, -1))
            cc = (cc_rgb + cc_ir) / 2.0
            # print(cc.shape)

            images_yir = images_yir.cpu().numpy()
            reconstruct = reconstruct.cpu().numpy()
            y_1=y_1.cpu().numpy()
            ir_1 = ir_1.cpu().numpy()
            ir = ir.cpu().numpy()
            # y_label = y_label.cpu().numpy()
            # ir_label = ir_label.cpu().numpy()
            psnrrgb = PSNR(y_1, reconstruct)
            psnrir = PSNR(ir_1, reconstruct)
            psnr = (psnrrgb + psnrir) / 2.0
            print(psnr)
            # images_yir=images_yir.cpu().numpy()
            # reconstruct=reconstruct.cpu().numpy()
            # cc_rgb=np.corrcoef((y, reconstruct))
            # # print(images_yir[:,:1].shape)
            # cc_ir = np.corrcoef((ir, reconstruct))
            # cc=(cc_rgb+cc_ir)/2.0

            # print(reconstruct)
            # print(reconstruct.shape)
            loss = F.cross_entropy(logits, labels)
            acc = calculate_accuracy(logits, labels)
            # loss = 0
            # acc = 1
            psnr_avg += float(psnr)
            # cc_avg+=float(cc)
            ssim_avg += float(ssim)
            loss_avg += float(loss)
            acc_avg += float(acc)

            print('|- test iter %s/%s. loss: %.4f, acc: %.4f' \
                  % (it + 1, test_loader.n_iter, float(loss), float(acc)))
            # viz =label2rgb(logits,test_loader)
            # labellabel=label2rgb(labels,test_loader)
            # viz=viz[..., ::-1]
            # labellabel=labellabel[..., ::-1]
            # path=os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/outputforafnet/output/'+str(it)+'.png')
            # path2 = os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/outputforafnet/label/' + str(it) + '.png')
            # cv2.imwrite(path,viz)
            # cv2.imwrite(path2,labellabel)
            # boundary1=F.sigmoid(boundary_logits1)
            # boundary2=F.sigmoid(boundary_logits2)
            filepath = '/content/drive/MyDrive/MFNet-pytorch-master/weights/mffenet_reconrgbir/output2/' + names[
                0] + '.png'
            # filepath = '/content/drive/MyDrive/MFNet-pytorch-master/weights/mffenet_reconstruct/output11/' + names[0]+'.png'

            # predictions = logits.argmax(1)
            # visualize(filepath, predictions)
            visualize_reconstrucion(filepath, reconstruct)
            # ------------------------------可视化
            # saveimg=tensor2img(reconstruct)
            # cv2.imwrite(filepath,saveimg)

            # visualize_gray(filepath,reconstruct)

            # fpath=os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/outputforafnet/label/' + str(it) + '.png')
            # visualize_fusesegboundary(it, predictions,boundary1,boundary2)
            # visualize2(it,labels)
            # viz = label2rgb(predictions, test_loader)
            # labellabel = label2rgb(labels, test_loader)
            # viz = viz[..., ::-1]
            # labellabel = labellabel[..., ::-1]
            # path = os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/outputforafnet/output/' + str(it) + '.png')
            # path2 = os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/outputforafnet/label/' + str(it) + '.png')
            # cv2.imwrite(path, viz)
            # cv2.imwrite(path2, labellabel)
            # for gtcid in range(n_class):
            #     for pcid in range(n_class):
            #         gt_mask      = labels == gtcid
            #         pred_mask    = predictions == pcid
            #         intersection = gt_mask * pred_mask
            #         cf[gtcid, pcid] += int(intersection.sum())
    print('|-cc: %.4f, ssim: %.4f  ,psnr:%.4f  \n' \
          % (1 / test_loader.n_iter, ssim_avg / test_loader.n_iter, psnr_avg / test_loader.n_iter))
    # overall_acc, acc, IoU = calculate_result(cf)
    # recall_per_class = np.zeros(n_class)
    # for cid in range(0, n_class): # cid: class id
    #     if cf[cid, 0:].sum() == 0:
    #         recall_per_class[cid] = np.nan
    #     else:
    #         recall_per_class[cid] = float(cf[cid, cid]) / float(cf[cid, 0:].sum()) # recall (Acc) = TP/TP+FN
    # val_cls_recall = np.diag(cf) / (cf.sum(axis=1))
    #
    # val_recall = np.mean(val_cls_recall)
    # print('recall:',val_recall)
    # print('| recall:',recall_per_class.mean())
    # print('| overall accuracy:', overall_acc)
    # print('| accuracy of each class:', acc)
    # print('| class accuracy avg:', acc.mean())
    # print('| IoU:', IoU)
    # print('| class IoU avg:', IoU.mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test MFNet with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='mffenet_vgg')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()

    model_dir = os.path.join(model_dir, args.model_name)
    final_model_encoder_file = os.path.join(model_dir, 'encoderfinal.pth')
    final_model_rec_decoder_file = os.path.join(model_dir, 'rec_decoderfinal.pth')
    final_model_se_decoder_file = os.path.join(model_dir, 'se_decoderfinal.pth')
    best_model_encoder_file = os.path.join(model_dir, 'encoderbest.pth')
    best_model_rec_decoder_file = os.path.join(model_dir, 'rec_decoderbest.pth')
    best_model_se_decoder_file = os.path.join(model_dir, 'se_decoderbest.pth')
    # final_model_file = os.path.join(model_dir, 'final.pth')
    # best_model_file = os.path.join(model_dir, 'best.pth')
    # assert os.path.exists(best_model_file), 'model file `%s` do not exist' % (best_model_file)
    # assert os.path.exists(final_model_file), 'model file `%s` do not exist' % (final_model_file)

    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))

    main()
