import numpy as np
# import chainer
from PIL import Image
# from ipdb import set_trace as st
import torch

def calculate_accuracy(logits, labels):
    # inputs should be torch.tensor
    predictions = logits.argmax(1)
    no_count = (labels==-1).sum()
    predictions.data = predictions.data.long()
    labels.data = labels.data.long()
    count = ((predictions==labels)*(labels!=-1)).sum()
    acc = count.float() / (labels.numel()-no_count).float()
    return acc

def result(cf):
    col=np.sum(cf,axis=1) #每一行求和
    a=np.size(cf)
    for i in range(13):
        cf[i]=cf[i]/col[i]
    return cf

def l2_norm(input):
    """Perform l2 normalization operation on an input vector.
    code copied from liorshk's repository: https://github.com/liorshk/facenet_pytorch/blob/master/model.py
    """
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    alpha = 10
    output = output * alpha

    return output

def calculate_result(cf):

    # n_class = cf.shape[0]
    # conf = np.zeros((n_class, n_class))
    # IoU = np.zeros(n_class)
    # conf[:, 0] = cf[:, 0] / cf[:, 0].sum()
    # for cid in range(1, n_class):
    #     if cf[:, cid].sum() > 0:
    #         conf[:, cid] = cf[:, cid] / cf[:, cid].sum()
    #         IoU[cid] = cf[cid, cid] / (cf[cid, 1:].sum() + cf[1:, cid].sum() - cf[cid, cid])
    # overall_acc = np.diag(cf[1:, 1:]).sum() / cf[1:, :].sum()
    # acc = np.diag(conf)

    overall_acc = np.diag(cf).sum() / cf.sum()
    acc = np.diag(cf) / cf.sum(axis=1)
    IoU = np.diag(cf) / (cf.sum(axis=1) + cf.sum(axis=0) - np.diag(cf))

    return overall_acc, acc, IoU

def get_palette():
    #ir_seg
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

   # B
   #  car = [0, 0, 142],
   #  bike = [119, 11, 32],
   #  person = [220, 20, 60],
   #  sky = [70, 130, 180],
   #  tree = [107, 142, 35],
   #  traffic_lights = [250, 170, 30],
   #  road = [128, 64, 128],
   #  sidewalk = [244, 35, 232],
   #  building = [70, 70, 70],
   #  fence = [190, 153, 153],
   #  sign = [220, 220, 0],
   #  pole = [153, 153, 153],
   #  bus = [0, 60, 100]
   #  palette    = np.array([car,bike,person,sky,tree,traffic_lights,road,sidewalk,building,fence,sign,pole,bus])

   # #  # A
    # car = [0, 0, 142],
    # bike = [119, 11, 32],
    # person = [220, 20, 60],
    # sky = [70, 130, 180],
    # tree = [107, 142, 35],
    # grass = [152, 251, 152],
    # road = [128, 64, 128],
    # sidewalk = [244, 35, 232],
    # building = [70, 70, 70],
    # fence = [190, 153, 153],
    # sign = [220, 220, 0],
    # pole = [153, 153, 153],
    # block = [180, 165, 180]
    # palette = np.array([car, bike, person, sky, tree, grass, road, sidewalk, building, fence, sign, pole, block])

    # unlabelled = [0,0,0]
    # car = [0, 0, 142],
    # bike = [119, 11, 32],
    # person = [220, 20, 60],

    # unlabelled = [0,0,0]
    # car        = [64,0,128]
    # bike       = [0,128,192]
    # person     = [64,64,0]
    # palette = np.array([unlabelled, car, bike, person])

    # unlabelled = [0, 0, 0]
    # car = [0, 0, 142],
    # bike = [119, 11, 32],
    # person = [220, 20, 60],
    # sky = [70, 130, 180],
    # tree = [107, 142, 35],
    # road = [128, 64, 128],
    # sidewalk = [244, 35, 232],
    # building = [70, 70, 70],
    # fence = [190, 153, 153],
    # sign = [220, 220, 0],
    # pole = [153, 153, 153],
    # palette    = np.array([unlabelled,car,bike,person,sky,tree,road,sidewalk,building,fence,sign,pole])

    return palette
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def visualize_reconstrucion(names, reconstruction):

    for (i, pred) in enumerate(reconstruction):
        print(np.max(pred))
        print(np.min(pred))
        print(pred.shape)
        c,h,w=pred.shape
        pred=normalization(pred)
        print(np.max(pred))
        print(np.min(pred))
        pred = ((pred))* 255
        if c<3:

            pred=pred.squeeze(0)
        print(pred.shape)
        img = Image.fromarray(np.uint8(pred))
        # img = Image.fromarray(np.uint8(pred.transpose(1,2,0))).convert('RGB')
        print(names)
        img.save(names)
def visualize(names, predictions):
    palette = get_palette()
    # palette = get_palette2()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(1, int(predictions.max())):  #for RGB-IR
        # for cid in range(0, int(predictions.max()) + 1):     # for A/B
            img[pred == cid] = palette[cid]

        img = Image.fromarray(np.uint8(img))
        # img = Image.fromarray(np.uint8(img[10:310, 8:408]))
        # img = Image.fromarray(np.uint8(img[26:326, 24:424]))
        # img.save(names[i].replace('.png', '_pred.png'))
        img.save(names)

def visualize_gray(names, predictions):
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].squeeze(0).cpu().numpy()* 255
        print(pred.shape)
        print(np.max(pred))
        img = Image.fromarray(pred)
        # img = Image.fromarray(np.uint8(img[10:310, 8:408]))
        # img = Image.fromarray(np.uint8(img[26:326, 24:424]))
        # img.save(names[i].replace('.png', '_pred.png'))
        # img = img.resize((400, 300), Image.BILINEAR)
        img.convert('L').save(names)

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    #step1：首先通过squeez()将输入的tensor去掉batch这个维度,变为[c,h,w],
    #之后转为float和cpu，再将tensor的值限制在[0,1]之间（有时候tensor为负值）
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    #step1：将tensor进行一个线性拉伸，拉伸到最大值为1，最小值为0
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        # n_img = len(tensor)
        # img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        # img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        print(1)
    elif n_dim == 3:  #一般情况下会直接跳到这里
        img_np = tensor.numpy()  #转为array形式
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # CHW->HWC, RGB->BGR（因为后续要用cv2保存）
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()  #乘以255转为uint8
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)
