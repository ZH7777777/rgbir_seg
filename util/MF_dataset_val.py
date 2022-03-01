# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from util.augmentation import RandomFlip,RandomCrop3

# from ipdb import set_trace as st


class MF_dataset_val(Dataset):

    def __init__(self, data_dir, split,max_iters, have_label, input_h=480, input_w=640 ,transform=[]):
        super(MF_dataset_val, self).__init__()

        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.is_train  = have_label
        if not max_iters == None:
            self.names = self.names * int(np.ceil(float(max_iters) / len(self.names)))
        self.n_data    = len(self.names)


    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image     = np.asarray(Image.open(file_path)) # (w,h,c)#(hwc)
        # image.flags.writeable = True
        return image

    def get_train_item(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'images')
        label = self.read_image(name, 'labels')

        for func in self.transform:
            image, label = func(image, label)
        # image,label=RandomFlip()(image,label)
        # image=Image.fromarray(image)
        # label=Image.fromarray(label)
        # image,label=RandomCrop3((240,320))(image,label)
        image=np.asarray(image)
        label=np.asarray(label)

        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255
        label = np.asarray(Image.fromarray(label).resize((self.input_w, self.input_h)), dtype=np.int64)

        return torch.tensor(image), torch.tensor(label), name

    def get_test_item(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'images')
        image = np.asarray(Image.fromarray(image).resize((self.input_h, self.input_w)), dtype=np.float32).transpose((2,0,1))/255

        return torch.tensor(image), name


    def __getitem__(self, index):

        if self.is_train is True:
            return self.get_train_item(index)
        else: 
            return self.get_test_item (index)

    def __len__(self):
        return self.n_data

if __name__ == '__main__':
    # data_dir = 'E:\google drive/ir_seg_dataset2/'
    data_dir = '/content/drive/MyDrive/ir_seg_dataset2'
    MF_dataset()
