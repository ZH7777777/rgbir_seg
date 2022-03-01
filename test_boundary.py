# coding:utf-8
import os
import argparse
import time
import numpy as np
# from thop import profile
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from util.MF_dataset import MF_dataset
from util.util import calculate_accuracy, calculate_result

from model import FuseSeguseboundary,mffenetusecswintrans,fusesegusetrans_new,fusesegusetrans5,fusesegusetrans2,MFNet,RTFNet,afnet,FuseSeg,FuNNet,afnetusetrans,fusesegusetrans,fusesegusetrans3,fusesegusetrans4
from train_boundary import n_class, data_dir, model_dir
from util.util import label2rgb
from util.util import visualize,visualize2,visualize_fusesegboundary
import cv2



def main():
    
    cf = np.zeros((n_class, n_class))

    model = eval(args.model_name)(n_class=n_class)
    if args.gpu >= 0: model.cuda(args.gpu)
    print('| loading model file %s... ' % final_model_file, end='')
    # model.load_state_dict(torch.load(final_model_file, map_location={'cuda:0':'cuda:1'}))
    model.load_state_dict(torch.load(best_model_file))
    print('done!')


    test_dataset  = MF_dataset(data_dir, 'test', have_label=True)
    test_loader  = DataLoader(
        dataset     = test_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    test_loader.n_iter = len(test_loader)

    loss_avg = 0.
    acc_avg  = 0.
    model.eval()
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
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images)
            labels = Variable(labels)
            if args.gpu >= 0:
                images = images.cuda(args.gpu)
                labels = labels.cuda(args.gpu)

            logits,boundary_logits1,boundary_logits2 = model(images)
            loss = F.cross_entropy(logits, labels)
            acc = calculate_accuracy(logits, labels)
            loss_avg += float(loss)
            acc_avg  += float(acc)

            print('|- test iter %s/%s. loss: %.4f, acc: %.4f' \
                    % (it+1, test_loader.n_iter, float(loss), float(acc)))
            # viz =label2rgb(logits,test_loader)
            # labellabel=label2rgb(labels,test_loader)
            # viz=viz[..., ::-1]
            # labellabel=labellabel[..., ::-1]
            # path=os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/outputforafnet/output/'+str(it)+'.png')
            # path2 = os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/outputforafnet/label/' + str(it) + '.png')
            # cv2.imwrite(path,viz)
            # cv2.imwrite(path2,labellabel)
            boundary1=F.sigmoid(boundary_logits1)
            boundary2=F.sigmoid(boundary_logits2)

            predictions = logits.argmax(1)
            # fpath=os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/outputforafnet/label/' + str(it) + '.png')
            visualize_fusesegboundary(it, predictions,boundary1,boundary2)
            # visualize2(it,labels)
            # viz = label2rgb(predictions, test_loader)
            # labellabel = label2rgb(labels, test_loader)
            # viz = viz[..., ::-1]
            # labellabel = labellabel[..., ::-1]
            # path = os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/outputforafnet/output/' + str(it) + '.png')
            # path2 = os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/outputforafnet/label/' + str(it) + '.png')
            # cv2.imwrite(path, viz)
            # cv2.imwrite(path2, labellabel)
            for gtcid in range(n_class): 
                for pcid in range(n_class):
                    gt_mask      = labels == gtcid 
                    pred_mask    = predictions == pcid
                    intersection = gt_mask * pred_mask
                    cf[gtcid, pcid] += int(intersection.sum())

    overall_acc, acc, IoU = calculate_result(cf)

    print('| overall accuracy:', overall_acc)
    print('| accuracy of each class:', acc)
    print('| class accuracy avg:', acc.mean())
    print('| IoU:', IoU)
    print('| class IoU avg:', IoU.mean())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test MFNet with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='FuseSeguseboundary')
    parser.add_argument('--batch_size',  '-B',  type=int, default=2)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=8)
    args = parser.parse_args()

    model_dir        = os.path.join(model_dir, args.model_name)
    final_model_file = os.path.join(model_dir, 'final.pth')
    best_model_file = os.path.join(model_dir, 'best.pth')
    # assert os.path.exists(best_model_file), 'model file `%s` do not exist' % (best_model_file)
    # assert os.path.exists(final_model_file), 'model file `%s` do not exist' % (final_model_file)

    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))

    main()
