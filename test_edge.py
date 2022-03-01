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
from util.MF_dataset_edge import MF_dataset_edge
from util.MF_dataset import MF_dataset
from util.util import calculate_accuracy, calculate_result

from model import mffenet,BDCN,FuseSeguseedgenet,FuseSeguseuncertainty,mffenetusetrans_c,FuseSeguseboundary,mffenetusecswintrans,fusesegusetrans_new,fusesegusetrans5,fusesegusetrans2,MFNet,RTFNet,afnet,FuseSeg,FuNNet,afnetusetrans,fusesegusetrans,fusesegusetrans3,fusesegusetrans4
from train_edge import n_class, data_dir, model_dir
from util.util import label2rgb
# from util.util import visualize,visualize2,visualize_fusesegboundary
from util.utilvisualize import visualize
import cv2
import util.joint_transforms as transforms



def main():
    image_testtransform = transforms.Compose(

        [

            transforms.ToTensor(220)])
    cf = np.zeros((n_class, n_class))

    model = eval(args.model_name)(n_class=n_class)
    # edgenet=BDCN(rate=4)
    if args.gpu >= 0:
        model.cuda(args.gpu)
        # edgenet.cuda(args.gpu)
    print('| loading model file %s... ' % final_model_file, end='')
    # model.load_state_dict(torch.load(final_model_file, map_location={'cuda:0':'cuda:1'}))
    model.load_state_dict(torch.load(best_model_file))
    # edgenet.load_state_dict(torch.load(edgenet_model_file))
    # edgenet
    print('done!')


    test_dataset  = MF_dataset_edge(data_dir, 'test', have_label=True)
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
    # edgenet.eval()
    # #parameters-------------------------------------------
    input=torch.randn(1,4,480,640).cuda()
    macs,params=profile(model,inputs=(input,))
    print("macs",macs)
    print("params",params)
    # #-------------------------------------------------------
    # #speed test--------------------------------------------
    time_spent=[]
    iteration=100
    for _ in range(iteration):
        t_start=time.time()
        with torch.no_grad():
            model(input)
        torch.cuda.synchronize()
        time_spent.append(time.time()-t_start)
    elapsed_time=np.mean(time_spent)
    print('Elapsed time:[%.2f s/%d iter]'%(elapsed_time,iteration))
    print('speed time:%.4f s/iter    fps:%.2f'%(elapsed_time,1/elapsed_time))
    # #-----------------------------------------------------------------------------
    with torch.no_grad():
        for it, (images, labels,edge,names) in enumerate(test_loader):
            images = Variable(images)
            labels = Variable(labels)
            # boundary_label=Variable(boundary_label)
            if args.gpu >= 0:
                images = images.cuda(args.gpu)
                labels = labels.cuda(args.gpu)
                # boundary_label=boundary_label.cuda(args.gpu)
            epo=1
            # ir_input = images[:, 3:]
            # ir4edge = torch.cat((ir_input, ir_input, ir_input), dim=1)
            # edge = edgenet(ir4edge)
            # edge_out = F.sigmoid(edge[-1])  # 2,1,480,640
            # print(np.max(edge_out.detach().cpu().numpy()))
            # print(edge_out)
            # save_dir=os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/weights/FuseSeguseedgenet/edge'+'1561.png')
            # for i in range(2):
            #     visual=edge_out[i].cpu().data.numpy()[ 0, :, :]
            #     # visual=visual*255
            #     # print(visual)
            #     cv2.imwrite(save_dir, visual)
            logits, edge_output = model(images)
            # for i in range(2):
            #     edge_view=F.sigmoid(edge_output[i])
            #     edge_view = edge_view.cpu().data.numpy()[ 0, :, :]
            #     save_dir = os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/weights/FuseSeguseedgenet/edge2/'+str(names[i])+'.png')
            #     cv2.imwrite(save_dir, 255 * edge_view)




            # logits = model(images, labels, edge_out, epo)


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
            # boundary1=F.sigmoid(boundary_logits1)
            # boundary2=F.sigmoid(boundary_logits2)

            predictions = logits.argmax(1)
            fpath = os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/weights/mffenet_edge/mffenet/output/' + names[0] + '.png')
            # visualize(fpath, predictions)
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
    parser.add_argument('--model_name',  '-M',  type=str, default='mffenet')
    parser.add_argument('--batch_size',  '-B',  type=int, default=1)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=8)
    args = parser.parse_args()

    model_dir        = os.path.join(model_dir, args.model_name)
    final_model_file = os.path.join(model_dir, 'final.pth')
    edge_final_model_file = os.path.join(model_dir, 'edge_final.pth')
    best_model_file = os.path.join(model_dir, 'best.pth')
    edge_best_model_file = os.path.join(model_dir, 'edge_best.pth')
    edgenet_model_file = os.path.join(model_dir, 'bdcn_pretrained_on_bsds500.pth')
    tmp_model_file = os.path.join(model_dir, 'tmp.pth')
    # assert os.path.exists(best_model_file), 'model file `%s` do not exist' % (best_model_file)
    # assert os.path.exists(final_model_file), 'model file `%s` do not exist' % (final_model_file)

    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))

    main()
