import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
from PIL import Image
import json

default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

class iclevr_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        self.root = args.data_root
        self.mode = mode
        if mode == 'train':
            with open(args.train_json, 'r') as file:
                self.data = list(json.load(file).items())
        elif mode == 'test':
            with open(args.test_json, 'r') as file:
                self.data = json.load(file)
        elif mode == 'new_test':
            with open(args.new_test_json, 'r') as file:
                self.data = json.load(file)
        with open('../dataset/objects.json', 'r') as file:
            self.object_dict = json.load(file)
        self.cls = len(self.object_dict)
                
    def __len__(self):
        return len(self.data)
        
    def get_img(self, index):
        fname = f'{self.root}/{self.data[index][0]}'
        image = torchvision.io.read_image(fname).to(torch.float32)[:3]
        size = min(image.shape[1:])
        image = image / 255.
        transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(64, antialias=True)
        ])
        return transform(image)
    
    def get_cond(self, index):
        cond = self.data[index][1] if self.mode == 'train' else self.data[index]
        one_hot_cond = torch.zeros(self.cls)
        for label in cond:
            one_hot_cond[self.object_dict[label]] = 1.0
        return one_hot_cond

    def __getitem__(self, index):
        if self.mode == 'train':
            img = self.get_img(index)
            cond =  self.get_cond(index)
            return img, cond
        else:
            cond =  self.get_cond(index)
            return cond
