import pandas as pd
from torch.utils import data
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('./data/train_img.csv')
        label = pd.read_csv('./data/train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('./data/test_img.csv')
        label = pd.read_csv('./data/test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        
        path = self.root + self.img_name[index] + '.jpeg'
        label = torch.Tensor(np.zeros(5,))
        label[self.label[index]] = 1
        img = torchvision.io.read_image(path).to(torch.float32)
        size = min(img.shape[1:])
        img = img / 255
        if self.mode == 'test':
            transform = transforms.Compose([
                transforms.CenterCrop(size),
                transforms.Resize(512, antialias=True)
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomCrop(size),
                transforms.Resize(512, antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(180)
            ])
        img = transform(img.float())
        
        return img, label