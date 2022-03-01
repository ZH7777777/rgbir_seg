# coding:utf-8
import numpy as np
# import chainer
from PIL import Image
from torch.optim import lr_scheduler
# from ipdb import set_trace as st
import cv2
import os
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(index, classes):
    size = index.size()[:1] + (classes,)
    view = index.size()[:1] + (1,)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.
    return mask.scatter_(1, index, ones)


class StaticLoss(nn.Module):
    def __init__(self, num_classes=9, gamma=1.0, eps=1e-7, size_average=True, one_hot=True, ignore=255, weight=None):
        super(StaticLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.classs = num_classes
        self.size_average = size_average
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.ignore = ignore
        self.weights = weight
        self.raw = False
        if (num_classes < 9):
            self.raw = True

    def forward(self, input, target, eps=1e-5):
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C

        if self.raw:
            target_left, target_right, target_up, target_down = target, target, target, target
            target_left[:, :-1, :] = target[:, 1:, :]
            target_right[:, 1:, :] = target[:, :-1, :]
            target_up[:, :, 1:] = target[:, :, :-1]
            target_down[:, :, :-1] = target[:, :, 1:]
            target_left, target_right, target_up, target_down = target_left.view(-1), target_right.view(-1), target_up.view(-1), target_down.view(-1)
            target_left2, target_right2, target_up2, target_down2 = target, target, target, target
            target_left2[:, :-1, 1:] = target[:, 1:, :-1]
            target_right2[:, 1:, 1:] = target[:, :-1, :-1]
            target_up2[:, 1:, :-1] = target[:, :-1, 1:]
            target_down2[:, :-1, :-1] = target[:, 1:, 1:]
            target_left2, target_right2, target_up2, target_down2 = target_left2.view(-1), target_right2.view(-1), target_up2.view(-1), target_down2.view(-1)

        target = target.view(-1)
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]
            if self.raw:
                target_left, target_right, target_up, target_down = target_left[valid], target_right[valid], target_up[valid], target_down[valid]
                target_left2, target_right2, target_up2, target_down2 = target_left2[valid], target_right2[valid], target_up2[valid], target_down2[valid]

        if self.one_hot:
            target_onehot = one_hot(target, input.size(1))
            if self.raw:
                target_onehot2 = one_hot(target_left, input.size(1))+one_hot(target_right, input.size(1))\
                                 + one_hot(target_up, input.size(1))+one_hot(target_down, input.size(1)) \
                                 + one_hot(target_left2, input.size(1)) + one_hot(target_right2, input.size(1)) \
                                 + one_hot(target_up2, input.size(1)) + one_hot(target_down2, input.size(1))
                target_onehot = target_onehot+target_onehot2
                target_onehot[target_onehot > 1] = 1

        probs = F.softmax(input, dim=1)
        probs = (self.weights*probs * target_onehot).max(1)[0]
        probs = probs.clamp(self.eps, 1. - self.eps)
        log_p = probs.log()

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
def meta_update(model, meta_init, meta_init_grads, meta_alpha, meta_alpha_grads,
                meta_init_optimizer, meta_alpha_optimizer):
    # Unpack the list of grad dicts
    # init_gradients = {k: sum(d[k] for d in meta_init_grads) for k in meta_init_grads[0].keys()}
    # print(meta_init_grads)
    # print(len(meta_init_grads))
    # print(meta_init_grads.shape)
    # print(meta_init_grads.keys())
    # print(meta_init_grads[0].keys().shape)
    init_gradients = {k: (sum(d[k] for d in meta_init_grads) / len(meta_init_grads)) for k in meta_init_grads[0].keys()}
    # alpha_gradients = {k: sum(d[k] for d in meta_alpha_grads) for k in meta_alpha_grads[0].keys()}
    # alpha_gradients = {k: (sum(d[k] for d in meta_alpha_grads) / len(meta_init_grads)) for k in
    #                    meta_alpha_grads[0].keys()}
    # init_gradients = {k: (sum(d for d in meta_init_grads) / len(meta_init_grads)) for k in meta_init_grads.keys()}
    # # alpha_gradients = {k: sum(d[k] for d in meta_alpha_grads) for k in meta_alpha_grads[0].keys()}
    # alpha_gradients = {k: (sum(d for d in meta_alpha_grads) / len(meta_init_grads)) for k in
    #                    meta_alpha_grads.keys()}

    # dummy variable to mimic forward and backward
    dummy_x = Variable(torch.Tensor(np.random.randn(1)), requires_grad=False).cuda()

    # update meta_init(for initial weights)
    for k, init in meta_init.items():
        dummy_x = torch.sum(dummy_x * init)
    meta_init_optimizer.zero_grad()
    dummy_x.backward()
    for k, init in meta_init.items():
        init.grad = init_gradients[k]
    meta_init_optimizer.step()

    # update meta_alpha(for learning rate)
    # dummy_y = Variable(torch.Tensor(np.random.randn(1)), requires_grad=False).cuda()
    # for k, alpha in meta_alpha.items():
    #     dummy_y = torch.sum(dummy_y * alpha)
    # meta_alpha_optimizer.zero_grad()
    # dummy_y.backward()
    # for k, alpha in meta_alpha.items():
    #     alpha.grad = alpha_gradients[k]
    # meta_alpha_optimizer.step()
class probreweighting(nn.Module):
    def __init__(self, std=0.1,avg=1.0):
        super().__init__()
        self.std=std
        self.avg=avg
    def forward(self, preds,labels):
        preds=preds.cpu()
        labels=labels.cpu()
        for i in range(labels.size(0)):
            # labels=labels[i].flatten()


            # labels=labels.flatten()
            # mask = (label_true >= 0)
            hist = np.bincount(
                labels[i].flatten(),minlength=9)
            mask=hist>0
            hist2=hist[mask]
            # mean=np.mean(hist2)
            # arr_std=np.std(hist2)
            sum=np.sum(hist)
            hist=hist/sum
            hist=np.where(hist>0,-np.log(hist),0)
            # hist=-np.log(hist[mask])
            mean=np.mean(hist[mask])
            arr_std=np.std(hist[mask])
            a=((hist[mask] - mean) / arr_std)*self.std + self.avg
            weight=np.where(hist!=0,((hist-mean)/arr_std)*self.std+self.avg,1)
            for k in range(9):
                # print(preds[:,i].shape)
                # print(weight[i])
                # print(preds[i,k].size())
                a=preds[i,k]
                b=weight[k]
                preds[i,k]=preds[i,k]*weight[k]
            # logpt = -self.ce_fn(preds, labels)
            # pt = torch.exp(logpt)
            # loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return preds


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

# def getpalette():
#     return np.asarray(
#         [
#             # [0, 0, 0],
#             # [128, 0, 0], #褐红色
#             # [192, 192, 128],#淡黄色偏绿
#             # [128, 64, 128],#淡紫色
#             # [0, 0, 192],#蓝色
#     [0, 0, 0],
#     [64, 0, 128],
#     [64, 64, 0],
#     [0, 128, 192],
#     [0, 0, 192],
#     [128, 128, 0],
#     [64, 64, 128],
#     [192, 128, 128],
#     [192, 64, 0]
#
#         ]
#     )
def label2rgb(lbl, dataloader, img=None, n_labels=None, alpha=0.5):
    if n_labels is None:
        n_labels = lbl.max() + 1  # +1 for bg_label 0

    cmap = getpalette()
    # cmap = getpalette(n_labels)
    # cmap = np.array(cmap).reshape([-1, 3]).astype(np.uint8)
    lbl=lbl.data.cpu().numpy()

    lbl_viz = cmap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled

    if img is not None:

        # img_gray = skimage.color.rgb2gray(img)
        # img_gray = skimage.color.gray2rgb(img_gray)
        # img_gray *= 255
        lbl_viz = alpha * lbl_viz
        lbl_viz = lbl_viz.astype(np.uint8)

    return lbl_viz

class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=None):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input): 
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs
def get_scheduler(optimizer,args):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | poly | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':

        def lambda_rule(epoch):
            lr = 1.0 - max(0,
                           epoch + 1 - args.epochs) / float(args.niter_decay + 1)
            return lr

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'poly':

        def lambda_rule(epoch):
            lr = (1 - epoch / args.epoch_max)**args.lr_power
            return lr

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step, gamma=0.1)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            args, mode='min', factor=0.2, threshold=1e-4, patience=5)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_policy is None:
        scheduler = None
    else:
        return NotImplementedError(
            f'learning rate policy {args.lr_policy} is not implemented')
    return scheduler
def calculate_accuracy(logits, labels):
    # inputs should be torch.tensor
    predictions = logits.argmax(1)
    no_count = (labels==-1).sum()
    count = ((predictions==labels)*(labels!=-1)).sum()
    # if(count>predictions.numel()):
    #     print(predictions)
    #     print(labels)
    # print('predictionnumel:',predictions.numel())
    # print('no_count:',no_count)
    # print('labels.numel:',labels.numel())
    # print('count:',count)
    acc = count.float() / (labels.numel()-no_count).float()
    return acc

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)#每一行相加 tp/tp+fn
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc/recall : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
def calculate_result(cf):
    # n_class = cf.shape[0]
    # conf = np.zeros((n_class,n_class))
    # IoU = np.zeros(n_class)
    # conf[:,0] = cf[:,0]/cf[:,0].sum()
    # for cid in range(0,n_class):
    #     if cf[:,cid].sum() > 0:
    #         conf[:,cid] = cf[:,cid]/cf[:,cid].sum()
    #         IoU[cid]  = cf[cid,cid]/(cf[cid,1:].sum()+cf[1:,cid].sum()-cf[cid,cid])
    # overall_acc = np.diag(cf[1:,1:]).sum()/cf[1:,:].sum()
    # acc = np.diag(conf)
    overall_acc = np.diag(cf).sum() / cf.sum()
    acc = np.diag(cf) / cf.sum(axis=1)
    IoU = np.diag(cf) / (cf.sum(axis=1) + cf.sum(axis=0) - np.diag(cf))

    return overall_acc, acc, IoU


# for visualization
def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette


def visualize_fusesegboundary(it, predictions,boundary1,boundary2):
    palette = get_palette()

    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(1, int(predictions.max())):
            img[pred == cid] = palette[cid]
        path = os.path.join(
            '/content/drive/MyDrive/MFNet-pytorch-master/outputforfuseseguseboundary/semantic/' + str(it) + str(
                i) + '.png')
        print('path is %s' % path)
        img = img[..., ::-1]
        cv2.imwrite(path, img)
        pred = boundary1[i].squeeze().cpu().numpy()

        # print(pred.shape)
        pred = pred * 255
        print(pred)
        # img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        # for cid in range(1, int(predictions.max())):
        #     img[pred == cid] = palette[cid]
        path = os.path.join(
            '/content/drive/MyDrive/MFNet-pytorch-master/outputforfuseseguseboundary/boundary1/' + str(it) + str(i) + '.png')
        # print('path is %s'%path)
        img = pred
        # img= img[..., ::-1]
        cv2.imwrite(path, img)
        pred = boundary2[i].squeeze().cpu().numpy()

        # print(pred.shape)
        pred = pred * 255
        print(pred)
        # img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        # for cid in range(1, int(predictions.max())):
        #     img[pred == cid] = palette[cid]
        path = os.path.join(
            '/content/drive/MyDrive/MFNet-pytorch-master/outputforfuseseguseboundary/boundary2/' + str(it) + str(i) + '.png')
        # print('path is %s'%path)
        img = pred
        # img= img[..., ::-1]
        cv2.imwrite(path, img)


def visualize(it, predictions):
    palette = get_palette()

    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(1, int(predictions.max())):
            img[pred == cid] = palette[cid]
        path=os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/outputformffenetusetrans/semantic/' + str(it)+str(i) + '.png')
        print('path is %s'%path)
        img= img[..., ::-1]
        cv2.imwrite(path, img)
        # img = Image.fromarray(np.uint8(img))
        # img.save(names[i].replace('.png', '_pred.png'))
def visualize_salient(it, predictions):
    palette = get_palette()

    for (i, pred) in enumerate(predictions):
        pred = predictions[i].squeeze().cpu().numpy()
        
        print(pred.shape)
        pred=pred*255
        print(pred)
        # img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        # for cid in range(1, int(predictions.max())):
        #     img[pred == cid] = palette[cid]
        path=os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/outputformffenet/salient2/' + str(it)+str(i) + '.png')
        # print('path is %s'%path)
        img = pred
        # img= img[..., ::-1]
        cv2.imwrite(path, img)
        # img = Image.fromarray(np.uint8(img))
        # img.save(names[i].replace('.png', '_pred.png'))
def visualize_boundary(it, predictions):
    palette = get_palette()

    for (i, pred) in enumerate(predictions):
        pred = predictions[i].squeeze().cpu().numpy()
        pred=pred*255
        # img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        # for cid in range(1, int(predictions.max())):
        #     img[pred == cid] = palette[cid]
        path=os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/outputformffenet/boundary2/' + str(it)+str(i) + '.png')
        # print('path is %s'%path)
        img=pred
        # img= img[..., ::-1]
        cv2.imwrite(path, img)
        # img = Image.fromarray(np.uint8(img))
        # img.save(names[i].replace('.png', '_pred.png'))
def visualize2(it, predictions):
    palette = get_palette()

    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(1, int(predictions.max())):
            img[pred == cid] = palette[cid]
            # print(1)
        path=os.path.join('/content/drive/MyDrive/MFNet-pytorch-master/outputforrtfnet/label/' + str(it)+str(i) + '.png')
        img= img[..., ::-1]
        cv2.imwrite(path, img)
        # img = Image.fromarray(np.uint8(img))
        # img.save(names[i].replace('.png', '_pred.png'))

def visualize3(it, image,rgb,names):
    # palette = get_palette()
    for (i, pred) in enumerate(rgb):
        name=names[i]
        a = rgb[i].cpu().numpy()
        # images=images.permute(1,2,0)
        enhancement = a.transpose(1,2,0)
        # enhancement = img * mean_std[1] + mean_std[0]
        enhancement = (enhancement-enhancement.min())/(enhancement.max()-enhancement.min())
        enhancement = enhancement[:, :, ::-1]*255  # change to BGR

        # print(np.unique(img))
        # print(img.shape)
        # print(img)

        path = os.path.join(
            '/content/drive/MyDrive/MFNet-pytorch-master/lightnet/rgb/' + str(name) +'-'+'rgb'+ '.png')
        # print('path is %s' % path)
        # img = img[..., ::-1]
        cv2.imwrite(path, enhancement)

    for (i, pred) in enumerate(image):
        name = names[i]
        images = image[i].cpu().numpy()
        # images=images.permute(1,2,0)
        enhancement = images.transpose(1,2,0)
        # enhancement = img * mean_std[1] + mean_std[0]
        enhancement = (enhancement-enhancement.min())/(enhancement.max()-enhancement.min())
        enhancement = enhancement[:, :, ::-1]*255  # change to BGR

        # print(np.unique(img))
        # print(img.shape)
        # print(img)

        path = os.path.join(
            '/content/drive/MyDrive/MFNet-pytorch-master/lightnet/rgb_lightnet/' + str(name) +'-'+'rgb'+ '.png')
        # print('path is %s' % path)
        # img = img[..., ::-1]
        cv2.imwrite(path, enhancement)
def visualize4(it, image,ir,names):
    # palette = get_palette()
    for (i, pred) in enumerate(ir):
        name = names[i]
        images = ir[i].cpu().numpy()
        # images=images.permute(1,2,0)
        enhancement = images.transpose(1,2,0)
        # enhancement = img * mean_std[1] + mean_std[0]
        enhancement = (enhancement-enhancement.min())/(enhancement.max()-enhancement.min())
        enhancement = enhancement[:, :, ::-1]*255  # change to BGR

        # print(np.unique(img))
        # print(img.shape)
        # print(img)

        path = os.path.join(
            '/content/drive/MyDrive/MFNet-pytorch-master/lightnet/ir/' + str(name) +'-'+'ir'+ '.png')
        # print('path is %s' % path)
        # img = img[..., ::-1]
        cv2.imwrite(path, enhancement)


    for (i, pred) in enumerate(image):
        name = names[i]
        images =image[i].cpu().numpy()
        # img = images.transpose(1,2,0)*255
        # print(np.unique(img))
        enhancement = images.transpose(1, 2, 0)
        # enhancement = img * mean_std[1] + mean_std[0]
        enhancement = (enhancement - enhancement.min()) / (enhancement.max() - enhancement.min())
        enhancement = enhancement[:, :, ::-1] * 255  # change to BGR
    
        path = os.path.join(
            '/content/drive/MyDrive/MFNet-pytorch-master/lightnet/ir_lightnet/' + str(name) +'-'+'ir'+ '.png')
        # print('path is %s' % path)
        # img = img[..., ::-1]
        cv2.imwrite(path, enhancement)
if __name__ == '__main__':
    # with torch.no_grad():
        import torch
        import numpy
        # import os
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # cuda0 = torch.device('cuda:0')
        # # x = torch.rand((1, 4, 128, 128, 128), device=cuda0)
        # x = torch.rand((1, 4, 480,640))
        # model = afnetusetrans(n_class=9)
        # model.cpu()
        # y = model(x)
        y=torch.tensor([[2,2,2],
                  [4,5,6],
                  [7,8,0]])
        x=torch.ones(2,9,9,9)
        loss=probreweighting()(x,y)


        print(y.shape)