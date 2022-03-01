# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
from torchvision.transforms import functional as F
from torchvision.transforms import ToTensor
import cv2
# from ipdb import set_trace as st

def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

class Edge_dataset(Dataset):

    def __init__(self, data_dir, split, have_label, input_h=320, input_w=320 ,transform=[]):
        super(Edge_dataset, self).__init__()

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

    def get_edges(self, mask):
        edge = torch.ByteTensor(mask.size()).zero_()
        edge = edge.bool()
        edge[:, 1:] = edge[:, 1:] | (mask[:, 1:] != mask[:, :-1])
        edge[:, :-1] = edge[:, :-1] | (mask[:, 1:] != mask[:, :-1])
        edge[1:, :] = edge[1:, :] | (mask[1:, :] != mask[:-1, :])
        edge[:-1, :] = edge[:-1, :] | (mask[1:, :] != mask[:-1, :])
        return edge.float()

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        # image     = np.asarray(Image.open(file_path)) # (w,h,c)
        image = Image.open(file_path)
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
        label = np.asarray(label)

        # if self.transform:
        #     image, label = self.transform(image, label)
        for func in self.transform:
            image, label= func(image, label)


        # image = np.asarray(Image.fromarray(np.asarray(image)).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
        #     (2, 0, 1)) / 255
        # label = np.asarray(Image.fromarray(np.asarray(label)).resize((self.input_w, self.input_h), resample=Image.NEAREST),
        #                    dtype=np.int64)
        # image=torch.tensor(image)
        # label=torch.tensor(label)

        # image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))
        # image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255

        # label = np.asarray(Image.fromarray(label).resize((self.input_w, self.input_h)), dtype=np.int64)

        # edgemap = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # x = cv2.Sobel(edgemap, cv2.CV_16S, 1, 0)
        # y = cv2.Sobel(edgemap, cv2.CV_16S, 0, 1)
        # abx = cv2.convertScaleAbs(x)
        # aby = cv2.convertScaleAbs(y)
        # edgemap = cv2.addWeighted(abx, 0.5, aby, 0.5, 0)
        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h), Image.BILINEAR),
                           dtype=np.float32).transpose((2, 0, 1)) / 255
        # /
        label = np.asarray(Image.fromarray(label).resize((self.input_w, self.input_h), Image.NEAREST), dtype=np.int64)
        label = torch.from_numpy(np.array(label)).long()

        # image = ToTensor()(image)
        # label = torch.from_numpy(np.array(label)).long()
        edgemap = self.get_edges(label)
        edgemap = torch.unsqueeze(edgemap, dim=0)

        # _edgemap = mask_to_onehot(label, 13)

        # _edgemap = onehot_to_binary_edges(_edgemap, 2, 13)

        return torch.tensor(image), label, edgemap, name

    def get_test_item(self, index):
        name  = self.names[index]
        # image = self.read_image(name, 'images')
        image = self.read_image(name, 'RGB-IR')
        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255
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
