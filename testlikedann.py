import os
import argparse
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from util.MF_dataset import MF_dataset
from util.MF_dataset_val import MF_dataset_val
from util.util import calculate_accuracy, calculate_result

from model import fusesegusetrans2, MFNet, RTFNet, afnet, FuseSeg, FuNNet, afnetusetrans, fusesegusetrans, \
    fusesegusetrans3, fusesegusetrans4
from trainlikedann import n_class, data_dir, model_dir
from model.relight import LightNet
from util.util import label2rgb
from util.util import visualize, visualize2, visualize3, visualize4
import cv2


def main():
    cf = np.zeros((n_class, n_class))

    model = eval(args.model_name)(n_class=n_class)
    lightnet = LightNet()
    lightnet4ir = LightNet()

    model.to(args.device)
    lightnet.to(args.device)
    lightnet4ir.to(args.device)
    print('| loading model file %s... ' % final_model_file, end='')
    # model.load_state_dict(torch.load(final_model_file, map_location={'cuda:0':'cuda:1'}))
    model.load_state_dict(torch.load(final_model_file))
    lightnet.load_state_dict(torch.load(final_lightnet_file))
    lightnet4ir.load_state_dict(torch.load(final_lightnet4ir_file))
    # model.cuda()
    # lightnet.cuda()
    # lightnet4ir.cuda()
    print('done!')

    test_dataset = MF_dataset(data_dir, 'test', have_label=True)
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
    model.eval()
    lightnet.eval()
    lightnet4ir.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images)
            labels = Variable(labels)
            if args.gpu >= 0:
                # images = images.to(args.device)
                # labels = labels.to(args.device)
                images = images.cuda()
                labels = labels.cuda()
            # ------------------------------------lightnet------------------
            rgb = images[:, :3]
            ir = images[:, 3:]
            ir = torch.cat((ir, ir, ir), dim=1)  # b*3*h*w

            rgb_mean_light = rgb.mean()
            ir_mean_light = ir.mean()
            r = lightnet(rgb)
            r_ir = lightnet4ir(ir)
            enhanced_images_rgb = rgb + r
            enhanced_images_ir = ir + r_ir

            visualize3(it, r, enhanced_images_rgb, names)
            visualize4(it, r_ir, enhanced_images_ir, names)

            enhanced_images_ir = enhanced_images_ir[:, :1, :, :]

            # loss = 0.1 * enhanced_images_ir + 0.1 * enhanced_images_rgb
            # loss.backward()

            images = torch.cat((enhanced_images_rgb, enhanced_images_ir), dim=1)
            # --------------------------------------------------------------------------------
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            acc = calculate_accuracy(logits, labels)
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

            predictions = logits.argmax(1)
            # fpath=os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/outputforafnet/label/' + str(it) + '.png')
            # visualize(it, predictions)
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
                    gt_mask = labels == gtcid
                    pred_mask = predictions == pcid
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
    parser.add_argument('--model_name', '-M', type=str, default='FuseSeg')
    parser.add_argument('--batch_size', '-B', type=int, default=16)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    args = parser.parse_args()

    model_dir = os.path.join(model_dir, args.model_name)
    final_model_file = os.path.join(model_dir, 'final.pth')
    best_model_file = os.path.join(model_dir, 'best.pth')
    final_lightnet_file = os.path.join(model_dir, 'lightnetfinal.pth')
    best_lightnet_file = os.path.join(model_dir, 'lightnetbest.pth')
    final_lightnet4ir_file = os.path.join(model_dir, 'lightnet4irfinal.pth')
    best_lightnet4ir_file = os.path.join(model_dir, 'lightnet4irbest.pth')
    pre_lightnet_file = os.path.join(model_dir, 'dannet_deeplab_light.pth')

    # assert os.path.exists(best_model_file), 'model file `%s` do not exist' % (best_model_file)
    # assert os.path.exists(final_model_file), 'model file `%s` do not exist' % (final_model_file)

    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))

    main()