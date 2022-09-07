import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np

class PredictDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.img_list = os.listdir(self.dataset_dir)
        self.img_list.sort()

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def normalize(vol):
        vol = vol - np.min(vol)
        vol = vol / np.max(vol)
        return vol

    def load_nii(self, nii_dir):
        data = nib.load(nii_dir)
        affine = data.affine
        data = np.asarray(data.get_fdata(), dtype='float32')
        if data.shape == (512, 512, 512):
            data = data[::2, ::2, ::2]
        data = self.normalize(data)
        data = torch.tensor(data)
        data = data.unsqueeze(0)
        return data, affine


    def __getitem__(self, item):
        img_name = self.img_list[item]
        img_path = os.path.join(self.dataset_dir, img_name)
        img, affine = self.load_nii(img_path)

        data = {'img': img,
                'img_name': img_name,
                'affine': affine}

        return data
