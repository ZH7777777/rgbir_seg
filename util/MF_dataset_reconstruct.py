# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os.path as osp
import os
# from util.augmentation import RandomFlip,RandomCrop3
from .transform import *
# from ipdb import set_trace as st


class MF_dataset_reconstruct(Dataset):

    def __init__(self,rootpth, cropsize=(640, 480), mode='train',
    randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), *args, **kwargs):
        super(MF_dataset_reconstruct, self).__init__(*args, **kwargs)

        assert mode in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.mode = mode
        self.rootpath=rootpth
        print('self.mode', self.mode)
        self.ignore_lb = 255
        with open(os.path.join(rootpth, self.mode+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]
        self.n_data=len(self.names)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406,0.355), (0.229, 0.224, 0.225,0.071)),#ir通道的均值和方差由整个数据集的计算得出
        ])
        self.trans_train = Compose([
            # ColorJitter(
            #     brightness=0.5,
            #     contrast=0.5,
            #     saturation=0.5),
            HorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            # RandomScale(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            RandomCrop(cropsize)
        ])

    def rgb2ycbcr(self,rgb_image):
        """convert rgb into ycbcr"""
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            raise ValueError("input image is not a rgb image")
        rgb_image = rgb_image.astype(np.float32)
        # 1：创建变换矩阵，和偏移量
        transform_matrix = np.array([[0.257, 0.564, 0.098],
                                     [-0.148, -0.291, 0.439],
                                     [0.439, -0.368, -0.071]])
        shift_matrix = np.array([16, 128, 128])
        ycbcr_image = np.zeros(shape=rgb_image.shape)
        w, h, _ = rgb_image.shape
        # 2：遍历每个像素点的三个通道进行变换
        for i in range(w):
            for j in range(h):
                ycbcr_image[i, j, :] = np.dot(transform_matrix, rgb_image[i, j, :]) + shift_matrix
        return ycbcr_image

    def read_image(self, name, folder):
        # file_path = os.path.join(self.rootpath, '%s/%s.png' % (folder, name))
        # file_path = os.path.join(self.rootpath, '%s/%s' % (folder, name))
        file_path = os.path.join(self.rootpath, '%s/%s.png' % (folder, name))
        image     =Image.open(file_path) # (w,h,c)#(hwc)
        # image.flags.writeable = True
        return image

    def get_train_item(self, index):
        name  = self.names[index]
        # img = self.read_image(name, 'images')
        # y= self.read_image(name, 'y')
        # ir= self.read_image(name, 'cropinfrared')
        # y_y = self.read_image(name, 'train_y')
        y = self.read_image(name, 'ycbcr')
        # y=y[:,:,0]

        # img = self.read_image(name, 'ycbcr')
        label = self.read_image(name, 'labels')

        # salient_label = self.read_image(name, 'labels_salient')
        # boundary_label = self.read_image(name, 'labels_boundary')
        # rgb=np.asarray(img,dtype=np.float32)[:,:,:3]
        # ir = np.asarray(img,dtype=np.float32)[:,:,3:]
        # rgb = self.rgb2ycbcr(rgb)
        # image=np.concatenate((rgb,ir),axis=2)
        # img=Image.fromarray(image.astype('uint8'))

        if self.mode == 'train':

            im_lb = dict(im=y, lb=label)
            # im_lb = dict(im=y, lb=ir)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
            # y, ir = im_lb['im'], im_lb['lb']

        # img = self.to_tensor(img)
        # img=torch.tensor(np.asarray(img, dtype=np.float32).transpose((2,0,1))/255)
        # y_label = torch.tensor((np.asarray(y, dtype=np.float32) / 255) * 2 - 1).unsqueeze(0)
        # ir_label = torch.tensor((np.asarray(ir, dtype=np.float32)/ 255) * 2 - 1).unsqueeze(0)

        # img = self.to_tensor(y)
        # label = self.to_tensor(ir)
        img = self.to_tensor(img)
        label = np.asarray(label,
                           dtype=np.int64)
        label = torch.from_numpy(np.array(label)).long()
        # label = self.to_tensor(label)
        # img = torch.tensor((np.asarray(y, dtype=np.float32) / 255) * 2 - 1).unsqueeze(0)
        # label = torch.tensor((np.asarray(ir, dtype=np.float32) / 255) * 2 - 1).unsqueeze(0)

        # image_yir = torch.cat((img, label), dim=0)

        # img=self.to_tensor(img)
        # rgb=img[:3,:,:]
        # ir=img[3:,:,:]
        #
        #
        # y=rgb[:1,:,:]
        # image_yir=torch.cat((y,ir),dim=0)


        # label = np.array(label).astype(np.int64)
        # return image_yir,img, label, name
        # return image_yir,y_label,ir_label,name
        return img, label, name
    def get_test_item(self, index):
        name = self.names[index]
        # img = self.read_image(name, 'images')
        # y= self.read_image(name, 'y')
        # ir= self.read_image(name, 'cropinfrared')
        # y = self.read_image(name, 'train_y')
        # ir = self.read_image(name, 'ir')
        img = self.read_image(name, 'ycbcr')
        # img=img[:,:,0]

        # img = self.read_image(name, 'ycbcr')
        label = self.read_image(name, 'labels')

        # salient_label = self.read_image(name, 'labels_salient')
        # boundary_label = self.read_image(name, 'labels_boundary')
        # rgb=np.asarray(img,dtype=np.float32)[:,:,:3]
        # ir = np.asarray(img,dtype=np.float32)[:,:,3:]
        # rgb = self.rgb2ycbcr(rgb)
        # image=np.concatenate((rgb,ir),axis=2)
        # img=Image.fromarray(image.astype('uint8'))

        if self.mode == 'train':
            im_lb = dict(im=img, lb=label)
            # im_lb = dict(im=y, lb=ir)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
            # y, ir = im_lb['im'], im_lb['lb']

        # img = self.to_tensor(img)
        # img=torch.tensor(np.asarray(img, dtype=np.float32).transpose((2,0,1))/255)
        # y_label = torch.tensor((np.asarray(img, dtype=np.float32) / 255) * 2 - 1).unsqueeze(0)
        # ir_label = torch.tensor((np.asarray(ir, dtype=np.float32) / 255) * 2 - 1).unsqueeze(0)

        # img = self.to_tensor(y)
        # label = self.to_tensor(ir)
        img = self.to_tensor(img)
        label = np.asarray(label,
                           dtype=np.int64)
        label = torch.from_numpy(np.array(label)).long()
        # label = self.to_tensor(label)
        # img = torch.tensor((np.asarray(y, dtype=np.float32) / 255) * 2 - 1).unsqueeze(0)
        # label = torch.tensor((np.asarray(ir, dtype=np.float32) / 255) * 2 - 1).unsqueeze(0)

        # image_yir = torch.cat((img, label), dim=0)
        # y=self.to_tensor(y)
        # ir = self.to_tensor(ir)

        # img=self.to_tensor(img)
        # rgb=img[:3,:,:]
        # ir=img[3:,:,:]
        #
        #
        # y=rgb[:1,:,:]
        # image_yir=torch.cat((y,ir),dim=0)

        # label = np.array(label).astype(np.int64)
        # return image_yir,img, label, name
        # return image_yir,y_label,ir_label,name
        return img, label, name



    def __getitem__(self, index):

        if self.mode=='train':
            return self.get_train_item(index)
        if self.mode=='test':
            return self.get_test_item(index)



    def __len__(self):
        return self.n_data

if __name__ == '__main__':
    data_dir = 'E:\google drive\RoadScene/'
    cropsize = [64, 64]
    randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)
    # data_dir = '/content/drive/MyDrive/ir_seg_dataset2'
    # dataset=MF_dataset_reconstruct(rootpth=data_dir,cropsize=cropsize,mode='train', randomscale=randomscale)
    dataset= MF_dataset_reconstruct(rootpth=data_dir, mode='train', randomscale=randomscale)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    for it, (images_yir, names) in enumerate(train_loader):
        print(names)
        print(images_yir.shape)
