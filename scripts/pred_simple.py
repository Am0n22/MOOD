import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import nibabel as nib

from MOOD_dataset import PredictDataset
from models import AutoEncoderCov3DMem, UNet3D
from pytorch_ssim import ssim3D
###

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

def predict_folder_pixel(input_folder, target_folder, dataset_type):
    test_dir = input_folder
    output_dir = target_folder

    recon_ckpts = []
    seg_ckpts = []
    pred_shape = (256, 256, 256)
    if dataset_type == 'brain':
        seg_ckpts = ['/workspace/checkpoints/brain_seg_{}.pt'.format(i) for i in range(3)]
        recon_ckpts = ['/workspace/checkpoints/brain_recon_{}.pt'.format(i) for i in range(3)]
        pred_shape = (256, 256, 256)
    elif dataset_type == 'abdom':
        seg_ckpts = ['/workspace/checkpoints/abdom_seg_{}.pt'.format(i) for i in range(3)]
        recon_ckpts = ['/workspace/checkpoints/abdom_recon_{}.pt'.format(i) for i in range(3)]
        pred_shape = (512, 512, 512)
    else:
        print('wrong dataset type')

    test_dataset = PredictDataset(dataset_dir=test_dir)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 )

    test_names = os.listdir(test_dir)
    preds_by_pixel = {}
    for name in test_names:
        preds_by_pixel[name] = {'pred_img': np.zeros(pred_shape),
                                'affine': None}

    for i in range(3):
        recon_model_dir = recon_ckpts[i]
        seg_model_dir = seg_ckpts[i]

        recon_model = AutoEncoderCov3DMem(chnum_in=1, mem_dim=4096, shrink_thres=0.0001)
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
                img_test = data['img'].to(device)
                img_name = data['img_name'][0]
                img_affine = data['affine'][0]

                recon_res = recon_model(img_test)
                recon_img = recon_res['output']

                residual_map = torch.abs(img_test - recon_img)
                ssim_map = ssim3D(recon_img, img_test, size_average=False)
                seg_in = torch.cat([img_test, recon_img, residual_map, ssim_map], dim=1)

                seg_in = F.interpolate(seg_in, scale_factor=0.5, mode='nearest')
                seg_out = seg_model(seg_in)
                seg_out = F.interpolate(seg_out, scale_factor=2, mode='nearest')
                if dataset_type == 'abdom':
                    seg_out = F.interpolate(seg_out, scale_factor=2, mode='nearest')

                pred_img = tensor2numpy(seg_out)
                if preds_by_pixel[img_name]['affine'] is None:
                    preds_by_pixel[img_name]['affine'] = img_affine
                preds_by_pixel[img_name]['pred_img'] += pred_img

    for pred_name in preds_by_pixel.keys():
        pred_affine = preds_by_pixel[pred_name]['affine']
        pred_img = preds_by_pixel[pred_name]['pred_img'] / 3
        output_path = os.path.join(output_dir, pred_name)
        nib.Nifti1Image(pred_img, affine=pred_affine).to_filename(output_path)

def predict_folder_sample(input_folder, target_folder, dataset_type):
    test_dir = input_folder
    output_dir = target_folder

    recon_ckpts = []
    seg_ckpts = []
    pred_shape = (256, 256, 256)
    thres = 0
    if dataset_type == 'brain':
        seg_ckpts = ['/workspace/checkpoints/brain_seg_{}.pt'.format(i) for i in range(3)]
        recon_ckpts = ['/workspace/checkpoints/brain_recon_{}.pt'.format(i) for i in range(3)]
        pred_shape = (256, 256, 256)
        thres = 0.005
    elif dataset_type == 'abdom':
        seg_ckpts = ['/workspace/checkpoints/abdom_seg_{}.pt'.format(i) for i in range(3)]
        recon_ckpts = ['/workspace/checkpoints/abdom_recon_{}.pt'.format(i) for i in range(3)]
        pred_shape = (512, 512, 512)
        thres = 0.0035
    else:
        print('wrong dataset type')

    test_dataset = PredictDataset(dataset_dir=test_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_names = os.listdir(test_dir)
    preds_by_pixel = {}
    for name in test_names:
        preds_by_pixel[name] = {'pred_img': np.zeros(pred_shape),
                                'affine': None}

    for i in range(3):
        recon_model_dir = recon_ckpts[i]
        seg_model_dir = seg_ckpts[i]

        recon_model = AutoEncoderCov3DMem(chnum_in=1, mem_dim=4096, shrink_thres=0.0001)
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
                img_test = data['img'].to(device)
                img_name = data['img_name'][0]
                img_affine = data['affine'][0]

                recon_res = recon_model(img_test)
                recon_img = recon_res['output']

                residual_map = torch.abs(img_test - recon_img)
                ssim_map = ssim3D(recon_img, img_test, size_average=False)
                seg_in = torch.cat([img_test, recon_img, residual_map, ssim_map], dim=1)

                seg_in = F.interpolate(seg_in, scale_factor=0.5, mode='nearest')
                seg_out = seg_model(seg_in)
                seg_out = F.interpolate(seg_out, scale_factor=2, mode='nearest')
                if dataset_type == 'abdom':
                    seg_out = F.interpolate(seg_out, scale_factor=2, mode='nearest')

                pred_img = tensor2numpy(seg_out)
                if preds_by_pixel[img_name]['affine'] is None:
                    preds_by_pixel[img_name]['affine'] = img_affine
                preds_by_pixel[img_name]['pred_img'] += pred_img


    for pred_name in preds_by_pixel.keys():
        pred_img = preds_by_pixel[pred_name]['pred_img'] / 3
        abnomal_score = np.mean(pred_img) / (thres * 2)
        # if np.mean(pred_img) >= thres:
        #     abnomal_score = 1
        # else:
        #     abnomal_score = 0

        with open(os.path.join(target_folder, pred_name + ".txt"), "w") as write_file:
            write_file.write(str(abnomal_score))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-m", "--mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("-d", "--dataset", type=str, default="brain", help="can be either 'brain' or 'abdom'.", required=False)


    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    mode = args.mode
    dataset_type = args.dataset

    if mode == "pixel":
        predict_folder_pixel(input_dir, output_dir, dataset_type)
    elif mode == "sample":
        predict_folder_sample(input_dir, output_dir, dataset_type)
    else:
        print("Mode not correctly defined. Either choose 'pixel' oder 'sample'")
