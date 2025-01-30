from torch.utils import data
import os
from os.path import join
import numpy as np
import cv2
import random
from torchvision import transforms as transforms
from PIL import Image
import torch

def one_hot(labels, num_classes=-1):
    if num_classes == -1:
        num_classes = int(labels.max()) + 1
    one_hot_tensor = torch.zeros(labels.size() + (num_classes,), dtype=torch.int64)
    one_hot_tensor.scatter_(-1, labels.unsqueeze(-1).to(torch.int64), 1)
    return one_hot_tensor

def get_one_hot(labels, num_classes=-1):
    """用于分割网络的one hot"""
    labels = torch.as_tensor(labels)
    ones = one_hot(labels, num_classes)
    return ones.view(*labels.size(), num_classes)

class preheat_Loader(data.Dataset):

    def __init__(self, root='..', split='train', jitter_d = 0.5, jitter_p = 0.5, random_c = 0.2):
        self.dir = root
        self.train = False
        if split == 'train':
            self.img_list = os.listdir(root + '/images/training')
            self.img_dir = root + '/images/training'
            self.train = True
        elif split == 'test':
            self.img_list = os.listdir(root + '/images/test')
            self.img_dir = root + '/images/test'
        random.shuffle(self.img_list)

        color_jitter = transforms.ColorJitter(0.8 * jitter_d, 0.8 * jitter_d, 0.8 * jitter_d, 0.2 * jitter_d)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=jitter_p)
        self.random_c = random_c
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            rnd_color_jitter,
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))])
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_file = self.img_list[index]
        gt_file = img_file.replace('.jpg','.png')
        image = Image.open(join(self.img_dir, img_file)).convert('RGB')
        image = np.array(image)
        gt = np.array(cv2.imread(self.img_dir.replace('images','annotations') + '/' + gt_file, 0), dtype=np.int)
        if self.train:
            image = self.train_transform(image)
            gt_new = get_one_hot(gt, num_classes=4)
            gt_new = gt_new.permute(2, 0, 1).float()
            rot_k = random.randint(1, 4)
            filp = random.randint(1, 2)
            if filp == 1:
                image = torch.flip(image, dims=[1])
                gt_new = torch.flip(gt_new, dims=[1])
            image = torch.rot90(image, k=rot_k, dims=[1, 2])
            gt_new = torch.rot90(gt_new, k=rot_k, dims=[1, 2])
        else:
            image = self.test_transform(image)
            gt_new = get_one_hot(gt, num_classes=4)
            gt_new = gt_new.permute(2, 0, 1).float()
        return image, gt_new, img_file.split('.')[0].split('/')[-1]

class our_preheat_Loader(data.Dataset):
    def __init__(self, root='..', split='train', jitter_d = 0.5, jitter_p = 0.5, random_c = 0.2):
        self.dir = root
        self.train = False
        if split == 'train':
            self.img_list = os.listdir(root + '/images/training')
            self.img_dir = root + '/images/training'
            self.train = True
        elif split == 'test':
            self.img_list = os.listdir(root + '/images/test')
            self.img_dir = root + '/images/test'
        random.shuffle(self.img_list)

        color_jitter = transforms.ColorJitter(0.8 * jitter_d, 0.8 * jitter_d, 0.8 * jitter_d, 0.2 * jitter_d)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=jitter_p)
        self.random_c = random_c
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            rnd_color_jitter,
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))])
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_file = self.img_list[index]
        gt_file = img_file.replace('.jpg','.png')
        bou_file = img_file.replace('.jpg','.npy')
        image = Image.open(join(self.img_dir, img_file)).convert('RGB')
        image = np.array(image)
        gt = np.array(cv2.imread(self.img_dir.replace('images','annotations') + '/' + gt_file, 0), dtype=np.int)
        if self.train:
            r1 = 1 - random.uniform(0, self.random_c)
            r2 = 1 - random.uniform(0, self.random_c)
            r3 = random.uniform(0, 1 - r1)
            r4 = random.uniform(0, 1 - r2)
            label_bou = np.load(self.img_dir.replace('images', 'annotations').replace('training', 'boundary') + '/' + bou_file)
            label_bou = torch.tensor(label_bou)
            image = self.train_transform(image)
            gt_new = get_one_hot(gt, num_classes=4)
            gt_new = gt_new.permute(2, 0, 1).float()
            rot_k = random.randint(1, 4)
            filp = random.randint(1, 2)
            if filp == 1:
                image = torch.flip(image, dims=[1])
                gt_new = torch.flip(gt_new, dims=[1])
                label_bou = torch.flip(label_bou, dims=[1])
            image = torch.rot90(image, k=rot_k, dims=[1, 2])
            gt_new = torch.rot90(gt_new, k=rot_k, dims=[1, 2])
            label_bou = torch.rot90(label_bou, k=rot_k, dims=[1, 2])
            _, h, w = image.shape
            image = image[:, int(r3 * h):int(r3 * h + r1 * h), int(r4 * w):int(r4 * w + r2 * w)]
            image = transforms.Resize([h, w])(image)
            gt_new = gt_new[:, int(r3 * h):int(r3 * h + r1 * h), int(r4 * w):int(r4 * w + r2 * w)]
            gt_new = transforms.Resize([h, w])(gt_new)
            label_bou = label_bou[:, int(r3 * h):int(r3 * h + r1 * h), int(r4 * w):int(r4 * w + r2 * w)]
            label_bou = transforms.Resize([h, w])(label_bou)
            return image, gt_new, label_bou, img_file.split('.')[0].split('/')[-1]
        else:
            image = self.test_transform(image)
            gt_new = get_one_hot(gt, num_classes=4)
            gt_new = gt_new.permute(2, 0, 1).float()
            return image, gt_new, img_file.split('.')[0].split('/')[-1]


class our_preheat_Loader_noaug(data.Dataset):
    def __init__(self, root='..', split='train', jitter_d = 0.5, jitter_p = 0.5, random_c = 0.2):
        self.dir = root
        self.train = False
        jitter_d = 0
        jitter_p = 0
        random_c = 0
        if split == 'train':
            self.img_list = os.listdir(root + '/images/training')
            self.img_dir = root + '/images/training'
            self.train = True
        elif split == 'test':
            self.img_list = os.listdir(root + '/images/test')
            self.img_dir = root + '/images/test'
        random.shuffle(self.img_list)

        color_jitter = transforms.ColorJitter(0.8 * jitter_d, 0.8 * jitter_d, 0.8 * jitter_d, 0.2 * jitter_d)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=jitter_p)
        self.random_c = random_c
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            rnd_color_jitter,
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))])
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_file = self.img_list[index]
        gt_file = img_file.replace('.jpg','.png')
        bou_file = img_file.replace('.jpg','.npy')
        image = Image.open(join(self.img_dir, img_file)).convert('RGB')
        image = np.array(image)
        gt = np.array(cv2.imread(self.img_dir.replace('images','annotations') + '/' + gt_file, 0), dtype=np.int)
        if self.train:
            r1 = 1 - random.uniform(0, self.random_c)
            r2 = 1 - random.uniform(0, self.random_c)
            r3 = random.uniform(0, 1 - r1)
            r4 = random.uniform(0, 1 - r2)
            label_bou = np.load(self.img_dir.replace('images', 'annotations').replace('training', 'boundary') + '/' + bou_file)
            label_bou = torch.tensor(label_bou)
            image = self.train_transform(image)
            gt_new = get_one_hot(gt, num_classes=4)
            gt_new = gt_new.permute(2, 0, 1).float()
            _, h, w = image.shape
            image = image[:, int(r3 * h):int(r3 * h + r1 * h), int(r4 * w):int(r4 * w + r2 * w)]
            image = transforms.Resize([h, w])(image)
            gt_new = gt_new[:, int(r3 * h):int(r3 * h + r1 * h), int(r4 * w):int(r4 * w + r2 * w)]
            gt_new = transforms.Resize([h, w])(gt_new)
            label_bou = label_bou[:, int(r3 * h):int(r3 * h + r1 * h), int(r4 * w):int(r4 * w + r2 * w)]
            label_bou = transforms.Resize([h, w])(label_bou)
            return image, gt_new, label_bou, img_file.split('.')[0].split('/')[-1]
        else:
            image = self.test_transform(image)
            gt_new = get_one_hot(gt, num_classes=4)
            gt_new = gt_new.permute(2, 0, 1).float()
            return image, gt_new, img_file.split('.')[0].split('/')[-1]