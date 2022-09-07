import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
train_dataset_id = 0  # 0 for brain, 1 for abdom

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
brain_dataset_origin_dir = '../dataset/brain_train/brain_train'
brain_dataset_seg_dir = '/media/lab426/My Passport1/MOOD/brain/img'
abdom_dataset_origin_dir = '../dataset/abdom_train/abdom_train'
abdom_dataset_seg_dir = '/media/lab426/My Passport1/MOOD/abdom/img'
brain_recon_dir = '/media/lab426/My Passport1/MOOD/brain/recon'
abdom_recon_dir = '/media/lab426/My Passport1/MOOD/abdom/recon'

recon_model_dir = '/home/lab426/mfr/MOOD/checkpoints/20220817_195926_model_final_pretraining_brain_memdim_4096_thres_0.0001_lr_0.0001'
recon_model_paths = []
for i in [29, 34, 39, 44, 45]:
    recon_model_paths.append(os.path.join(recon_model_dir, 'epoch_{}.pt'.format(i)))

if train_dataset_id == 0:
    dataset_origin_dir = brain_dataset_origin_dir
    dataset_seg_dir = brain_dataset_seg_dir
    recon_dir = brain_recon_dir
    dataset_name = 'brain'
elif train_dataset_id == 1:
    dataset_origin_dir = abdom_dataset_origin_dir
    dataset_seg_dir = abdom_dataset_seg_dir
    recon_dir = abdom_recon_dir
    dataset_name = 'abdom'
input_channel = 1
output_channel = 1

fold_num = 5
batch_size = 1
epoch_num = 50
num_workers = 4
learning_rate = 0.0001
resume = 0

entropy_loss_weight = 0.0005
sparse_shrink_thres = 0.0001
mem_dim_in = 4096

############ model saving dir path
saving_root = '/home/lab426/mfr/MOOD/checkpoints'
model_setting = 'newData_{}_seg_training_memdim_{}_thres_{}'.format(dataset_name, mem_dim_in, sparse_shrink_thres)
saving_model_path = os.path.join(saving_root, 'model_' + model_setting + '/')
utils.mkdir(saving_model_path)

local_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
blog_name = "{}_{}.csv".format(local_time, model_setting)
blog = open(os.path.join(saving_model_path, blog_name), "w")
blog_title = 'epoch,mse,entropy,dice'
blog.write(blog_title + '\n')

###### data
data_num = len(os.listdir(dataset_seg_dir))
id_all = list(range(data_num))

train_dataset = dataset.SegmentationDataset(dataset_origin_dir=dataset_origin_dir,
                                                         dataset_seg_dir=dataset_seg_dir,
                                                         ids=id_all)
print(len(train_dataset))
train_data_loader = DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers
                               )

# load pretraining parameters
recon_model = AutoEncoderCov3DMem(input_channel, mem_dim_in, shrink_thres=sparse_shrink_thres)
models = [torch.load(path) for path in recon_model_paths]
# worker_state_dict = [x.state_dict() for x in models]
worker_state_dict = models
weight_keys = list(worker_state_dict[0].keys())
fed_state_dict = collections.OrderedDict()
for key in weight_keys:
    key_sum = 0
    for i in range(len(models)):
        key_sum = key_sum + worker_state_dict[i][key]
    fed_state_dict[key] = key_sum / len(models)
recon_model.load_state_dict(fed_state_dict)

seg_model = UNet3D(in_channels=4, out_channels=1)

#########
device = torch.device("cuda")
recon_model.to(device)
seg_model.to(device)
recon_loss_func = nn.MSELoss().to(device)
entropy_loss_func = EntropyLossEncap().to(device)
dice_loss_func = Diceloss().to(device)

train_optimizer = torch.optim.Adam([
        {'params': recon_model.parameters(), 'lr': learning_rate},
        {'params': seg_model.parameters(), 'lr': learning_rate},
        ])

##
data_loader_len = len(train_data_loader)

for epoch_idx in range(resume, epoch_num):
    for batch_idx, data in enumerate(train_data_loader):
        img_origin = data['img_origin'].to(device)
        img_seg = data['img_seg'].to(device)
        mask = data['mask'].to(device)

        recon_res = recon_model(img_seg)
        recon_img = recon_res['output']
        att_w = recon_res['att']

        residual_map = torch.abs(img_origin - recon_img)
        ssim_map = ssim3D(recon_img, img_origin, size_average=False)
        seg_in = torch.cat([img_seg, recon_img, residual_map, ssim_map], dim=1)
        seg_in = torch.nn.functional.interpolate(seg_in, scale_factor=0.5, mode='nearest')
        # print(seg_in.shape)
        seg_out = seg_model(seg_in)
        seg_out = torch.nn.functional.interpolate(seg_out, scale_factor=2, mode='nearest')

        loss_recon = recon_loss_func(recon_img, img_origin)
        entropy_loss = entropy_loss_func(att_w)
        dice_loss = dice_loss_func(mask, seg_out)
        loss = loss_recon + entropy_loss * entropy_loss_weight + dice_loss

        blog_txt = '{},{:.6f},{:.6f},{:.6f}'.format(epoch_idx, loss_recon.item(), entropy_loss.item(),
                                                             dice_loss.item())
        # 'epoch,mse,entropy,dice'
        print(blog_txt)
        blog.write(blog_txt + '\n')

        ##
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # batch_size_ = img_origin.shape[0]
        # for i in range(batch_size_):
        #     print(recon_img.shape)
        #     recon_img_i = recon_img[i]
        #     recon_img_i = recon_img_i.cpu().detach().numpy()
        #     output_dir = os.path.join(recon_dir, str(fold), str(epoch_idx))
        #     if not os.path.exists(output_dir):
        #         os.makedirs(output_dir)
        #     output_path = os.path.join(output_dir, data['img_seg_name'][i])
        #     nib.Nifti1Image(recon_img_i, affine=data['affine'][i]).to_filename(output_path)


    torch.save(recon_model.state_dict(),
               os.path.join(saving_model_path, 'recon_epoch_{}.pt'.format(epoch_idx)))

    torch.save(seg_model.state_dict(),
               os.path.join(saving_model_path, 'seg_epoch_{}.pt'.format(epoch_idx)))
