# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from util.augmentation import RandomFlip,RandomCrop3
import torchvision.transforms as transforms

# from ipdb import set_trace as st


class MF_dataset_edge(Dataset):

    def __init__(self, data_dir, split, have_label, input_h=480, input_w=640 ,transform=[]):
        super(MF_dataset_edge, self).__init__()

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
        self.ColorJitte=transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5)
    def get_edges(self, mask):
        edge = torch.ByteTensor(mask.size()).zero_()
        edge = edge.bool()
        edge[:, 1:] = edge[:, 1:] | (mask[:, 1:] != mask[:, :-1])
        edge[:, :-1] = edge[:, :-1] | (mask[:, 1:] != mask[:, :-1])
        edge[1:, :] = edge[1:, :] | (mask[1:, :] != mask[:-1, :])
        edge[:-1, :] = edge[:-1, :] | (mask[1:, :] != mask[:-1, :])
        return edge.float()
    def get_binary(self, mask):
        binary = torch.ByteTensor(mask.size()).zero_()
        binary = binary.bool()
        binary[mask > 0] = 1

        return binary.float()

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image     = Image.open(file_path) # (w,h,c)#(hwc)
        # image.flags.writeable = True
        return image
    def read_label(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image     = np.asarray(Image.open(file_path)) # (w,h,c)#(hwc)
        # image.flags.writeable = True
        return image

    def get_train_item(self, index):
        name  = self.names[index]
        image = self.read_label(name, 'images')
        # image=self.ColorJitte(image)
        # image=np.asarray(image)
        label = self.read_label(name, 'labels')
        # salient_label = self.read_image(name, 'labels_salient')
        # boundary_label = self.read_image(name, 'labels_boundary')

        # image, label = self.transform(image,label)
        for func in self.transform:
            image, label= func(image, label)


        image=np.asarray(image,dtype=np.uint8)
        label=np.asarray(label,dtype=np.uint8)


        # image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255
        # label = np.asarray(Image.fromarray(label).resize((self.input_w, self.input_h),resample=Image.NEAREST), dtype=np.int64)
        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)),dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        label = np.asarray(Image.fromarray(label).resize((self.input_w, self.input_h), resample=Image.NEAREST),
                           dtype=np.int64)
        label = torch.from_numpy(np.array(label)).long()
        # label=torch.from_numpy(np.asarray(label)).long()
        binary_map=self.get_binary(label)


        edgemap = self.get_edges(label)
        edgemap = torch.unsqueeze(edgemap, dim=0)
        binary_map=torch.unsqueeze(binary_map, dim=0)

        # salient_label = np.asarray(Image.fromarray(salient_label).resize((self.input_w, self.input_h),resample=Image.NEAREST), dtype=np.int64)
        # boundary_label = np.asarray(Image.fromarray(boundary_label).resize((self.input_w, self.input_h),resample=Image.NEAREST), dtype=np.int64)

        return torch.tensor(image),label,edgemap,binary_map, name
        # return torch.tensor(image), label, edgemap, name
        # return image, label, edgemap, name

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
