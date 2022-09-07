import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
test_dataset_id = 1  # 0 for brain, 1 for abdom

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import dataset
import scipy.io as sio
import utils
import time
from models import AutoEncoderCov3DMem, UNet3D
from models import EntropyLossEncap, Diceloss
import random
from pytorch_ssim import *
from utils import *
import collections
from torch.cuda.amp import autocast as autocast

###
input_channel = 1
output_channel = 1

sparse_shrink_thres = 0.0001
mem_dim_in = 4096

test_dir = '/home/lab426/mfr/MOOD/dataset/abdom_toy/toy_cls/'
pred_shape = (512, 512, 512)
###### data
data_num = len(os.listdir(test_dir))
id_all = list(range(data_num))

test_dataset = dataset.PredictDataset(dataset_origin_dir=test_dir,
                                      ids=id_all)
print(len(test_dataset))
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             )


def tensor2numpy(img):
    if torch.is_tensor(img):
        if img.is_cuda:
            img = img.cpu().detach()
        img = img.numpy()
    if len(img.shape) == 5:
        img = img.squeeze(0).squeeze(0)
    elif len(img.shape) == 4:
        img = img.squeeze(0)
    else:
        if not len(img.shape) == 3:
            print('unrecognized tensor shape')
    return img


def save_nii(img_tensor, output_path, img_affine):
    img = tensor2numpy(img_tensor)
    nib.Nifti1Image(img, affine=img_affine).to_filename(output_path)


resume_ids_list = [
    # [0],
    # [1],
    # [2],
    # [3],
    # [4],
    # [5],
    # [6],
    # [7],
    # [8],
    # [9],
    # [10],
    # [11],
    # [12],
    # [13],
    # [14],
    # [15],

    [1,4,0]
]


for resume_ids in resume_ids_list:
    resume_ids_folder = 'ids_'
    for id in resume_ids:
        resume_ids_folder += str(id)
    output_dir = '/home/lab426/mfr/MOOD/dataset/abdom_toy/pred/' + resume_ids_folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ############ model saving dir path


    test_names = os.listdir(test_dir)
    test_names.sort()
    preds_by_pixel = {}
    for name in test_names:
        preds_by_pixel[name] = {'pred_img': np.zeros(pred_shape),
                                'affine': None}

    for resume_id in resume_ids:
        seg_model_dir = '/home/lab426/mfr/MOOD/checkpoints/model_last_Data_abdom_seg_training_memdim_4096_thres_0.0001/last_Data_abdom_seg_training_memdim_4096_thres_0.0001_seg_epoch_{}.pt'.format(
            resume_id)
        recon_model_dir = '/home/lab426/mfr/MOOD/checkpoints/model_last_Data_abdom_seg_training_memdim_4096_thres_0.0001/last_Data_abdom_seg_training_memdim_4096_thres_0.0001_recon_epoch_{}.pt'.format(
            resume_id)

        # load pretraining parameters
        recon_model = AutoEncoderCov3DMem(input_channel, mem_dim_in, shrink_thres=sparse_shrink_thres)
        recon_model_state_dict = torch.load(recon_model_dir)
        recon_model.load_state_dict(recon_model_state_dict)

        seg_model = UNet3D(in_channels=4, out_channels=1)
        seg_model_state_dict = torch.load(seg_model_dir)
        seg_model.load_state_dict(seg_model_state_dict)

        #########
        device = torch.device("cuda")
        recon_model.to(device)
        seg_model.to(device)

        with torch.no_grad():
            for data in test_dataloader:
                img_test = data['img_origin'].to(device)
                img_name = data['img_origin_name'][0]
                img_affine = data['affine'][0]

                recon_res = recon_model(img_test)
                recon_img = recon_res['output']

                residual_map = torch.abs(img_test - recon_img)
                ssim_map = ssim3D(recon_img, img_test, size_average=False)
                seg_in = torch.cat([img_test, recon_img, residual_map, ssim_map], dim=1)
                seg_in = torch.nn.functional.interpolate(seg_in, scale_factor=0.5, mode='nearest')
                # print(seg_in.shape)
                seg_out = seg_model(seg_in)
                seg_out = torch.nn.functional.interpolate(seg_out, scale_factor=2, mode='nearest')
                if test_dataset_id == 1:
                    seg_out = torch.nn.functional.interpolate(seg_out, scale_factor=2, mode='nearest')
                # save_nii(recon_img,
                #          output_path=os.path.join(output_dir, 'recon_'+img_name),
                #          img_affine=img_affine)
                # save_nii(residual_map,
                #          output_path=os.path.join(output_dir, 'residual_'+img_name),
                #          img_affine=img_affine)
                # save_nii(ssim_map,
                #          output_path=os.path.join(output_dir, 'ssim_'+img_name),
                #          img_affine=img_affine)
                pred_img = tensor2numpy(seg_out)
                if preds_by_pixel[img_name]['affine'] is None:
                    preds_by_pixel[img_name]['affine'] = img_affine
                preds_by_pixel[img_name]['pred_img'] += pred_img

    for pred_name in preds_by_pixel.keys():
        pred_affine = preds_by_pixel[pred_name]['affine']
        pred_img = preds_by_pixel[pred_name]['pred_img'] / len(resume_ids)
        # pred_img = np.where(pred_img >= 0.5, 1.0, 0.0)
        print(pred_name, np.mean(pred_img))
        final_nimg = nib.Nifti1Image(pred_img, affine=pred_affine)
        nib.save(final_nimg, os.path.join(output_dir, 'pred_'+pred_name))

