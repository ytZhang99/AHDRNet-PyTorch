import os
import cv2
import glob
import torch
import numpy as np

from option import args
from utils.utils import *
from torch.utils.data import Dataset


class TrainData(Dataset):
    def __init__(self, train_dir):
        self.train_dir = train_dir
        self.train_dir_list = os.listdir(self.train_dir)
        self.image_list = []
        self.crop_size = [args.patch_size, args.patch_size]
        for scene in range(len(self.train_dir_list)):
            exposure_file_path = os.path.join(self.train_dir, self.train_dir_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.train_dir, self.train_dir_list[scene]), args.ext[2])
            label_path = os.path.join(self.train_dir, self.train_dir_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

    def __getitem__(self, index):
        # Read exposure times in one scene
        expoTimes = ReadExpoTimes(self.image_list[index][0])
        # Read LDR image in one scene, BGR
        ldr_images = ReadImages(self.image_list[index][1])
        # Read HDR label, BGR
        label = ReadLabel(self.image_list[index][2])
        # ldr images process
        pre_img0 = LDR_to_HDR(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = LDR_to_HDR(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = LDR_to_HDR(ldr_images[2], expoTimes[2], 2.2)

        pre_img0 = np.concatenate((ldr_images[0], pre_img0), 2)
        pre_img1 = np.concatenate((ldr_images[1], pre_img1), 2)
        pre_img2 = np.concatenate((ldr_images[2], pre_img2), 2)

        # hdr label process
        label = range_compressor(label)

        H, W, _ = ldr_images[0].shape
        x = np.random.randint(0, H - self.crop_size[0] - 1)
        y = np.random.randint(0, W - self.crop_size[1] - 1)

        img0 = pre_img0[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)
        img1 = pre_img1[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)
        img2 = pre_img2[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)
        label = label[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)

        sample = {'input0': img0, 'input1': img1, 'input2': img2, 'label': label}

        return sample

    def __len__(self):
        return len(self.train_dir_list)


class ValData(Dataset):
    def __init__(self, val_dir):
        self.test_dir = val_dir
        self.test_dir_list = os.listdir(self.test_dir)
        self.image_list = []
        self.crop_size = [args.patch_size, args.patch_size]
        for scene in range(len(self.test_dir_list)):
            exposure_file_path = os.path.join(self.test_dir, self.test_dir_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.test_dir, self.test_dir_list[scene]), args.ext[2])
            label_path = os.path.join(self.test_dir, self.test_dir_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

    def __getitem__(self, index):
        # Read exposure times in one scene
        expoTimes = ReadExpoTimes(self.image_list[index][0])
        # Read LDR image in one scene, BGR
        ldr_images = ReadImages(self.image_list[index][1])
        # Read HDR label, BGR
        label = ReadLabel(self.image_list[index][2])
        # ldr images process
        pre_img0 = LDR_to_HDR(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = LDR_to_HDR(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = LDR_to_HDR(ldr_images[2], expoTimes[2], 2.2)

        pre_img0 = np.concatenate((ldr_images[0], pre_img0), 2)
        pre_img1 = np.concatenate((ldr_images[1], pre_img1), 2)
        pre_img2 = np.concatenate((ldr_images[2], pre_img2), 2)

        # hdr label process
        label = range_compressor(label)

        img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
        img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
        img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
        label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)

        sample = {'input0': img0, 'input1': img1, 'input2': img2, 'label': label}

        return sample

    def __len__(self):
        return len(self.test_dir_list)


class TestData(Dataset):
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.test_dir_list = os.listdir(self.test_dir)
        self.image_list = []
        self.crop_size = [args.patch_size, args.patch_size]
        for scene in range(len(self.test_dir_list)):
            exposure_file_path = os.path.join(self.test_dir, self.test_dir_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.test_dir, self.test_dir_list[scene]), args.ext[2])
            self.image_list += [[exposure_file_path, ldr_file_path]]

    def __getitem__(self, index):
        # Read exposure times in one scene
        expoTimes = ReadExpoTimes(self.image_list[index][0])
        # Read LDR image in one scene
        ldr_images = ReadImages(self.image_list[index][1])
        # ldr images process
        pre_img0 = LDR_to_HDR(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = LDR_to_HDR(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = LDR_to_HDR(ldr_images[2], expoTimes[2], 2.2)

        pre_img0 = np.concatenate((ldr_images[0], pre_img0), 2)
        pre_img1 = np.concatenate((ldr_images[1], pre_img1), 2)
        pre_img2 = np.concatenate((ldr_images[2], pre_img2), 2)

        img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
        img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
        img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)

        sample = {'input0': img0, 'input1': img1, 'input2': img2}

        return sample

    def __len__(self):
        return len(self.test_dir_list)