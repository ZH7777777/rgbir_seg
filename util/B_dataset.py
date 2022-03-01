# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import ToTensor
# from ipdb import set_trace as st


class B_dataset(Dataset):

    def __init__(self, data_dir, split, have_label, input_h=320, input_w=320 ,transform=[]):
        super(B_dataset, self).__init__()

        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.is_train  = have_label
        self.n_data    = len(self.names)


    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        # image     = np.asarray(Image.open(file_path))
        image     = Image.open(file_path)
        return image

    def get_train_item(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'RGB-IR')
        label = self.read_image(name, 'label_png')

        image = F.pad(image, (8, 10), fill=0, padding_mode='reflect')
        if self.split == 'train':
            label = F.pad(label, (8, 10), fill=255, padding_mode='reflect')
        elif self.split in ['val', 'test']:
            label = F.pad(label, (8, 10), fill=255)
        image = np.asarray(image)
        label=np.asarray(label)

        # pad
        # image = F.pad(image, (8, 10), fill=0)
        # label = F.pad(label, (8, 10), fill=255)

        # FuseSeg pad
        # image = F.pad(image, (24, 26), fill=0)
        # label = F.pad(label, (24, 26), fill=255)
        for func in self.transform:
            image, label= func(image, label)
        # if self.transform:
        #     image, label = self.transform(image, label)

        # image = ToTensor()(image)
        # label = torch.from_numpy(np.array(label)).long()
        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h), Image.BILINEAR), dtype=np.float32).transpose((2,0,1))/255
# /
        label = np.asarray(Image.fromarray(label).resize((self.input_w, self.input_h), Image.NEAREST), dtype=np.int64)
        label = torch.from_numpy(np.array(label)).long()

        # return torch.tensor(image), torch.tensor(label), name
        return torch.tensor(image), label, name

    def get_test_item(self, index):
        name  = self.names[index]
        # image = self.read_image(name, 'images')
        image = self.read_image(name, 'RGB-IR')
        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h), Image.BILINEAR), dtype=np.float32).transpose((2,0,1))/255
        # image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))

        return torch.tensor(image), name


    def __getitem__(self, index):

        if self.is_train is True:
            return self.get_train_item(index)
        else:
            return self.get_test_item (index)

    def __len__(self):
        return self.n_data

class RGB_dataset(Dataset):

    def __init__(self, data_dir, split, have_label, input_h=480, input_w=640 ,transform=[]):
        super(RGB_dataset, self).__init__()

        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.is_train  = have_label
        self.n_data    = len(self.names)

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image     = np.asarray(Image.open(file_path)) # (w,h,c)
        image.flags.writeable = True
        return image

    def get_train_item(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'VS')
        label = self.read_image(name, 'label_png')
        # image = self.read_image(name, 'images')
        # label = self.read_image(name, 'labels')

        for func in self.transform:
            image, label = func(image, label)

        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255
        label = np.asarray(Image.fromarray(label).resize((self.input_w, self.input_h)), dtype=np.int64)

        return torch.tensor(image), torch.tensor(label), name

    def get_test_item(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'VS')
        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255

        return torch.tensor(image), name

    def __getitem__(self, index):

        if self.is_train is True:
            return self.get_train_item(index)
        else:
            return self.get_test_item (index)

    def __len__(self):
        return self.n_data

if __name__ == '__main__':
    data_dir = '../../data/MF/'
    MF_dataset()
