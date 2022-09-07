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
from models import AutoEncoderCov3D, AutoEncoderCov3DMem
from models import EntropyLossEncap
import random
from pytorch_ssim import *
from utils import *
from torch.cuda.amp import autocast as autocast

import collections

###
brain_dataset_origin_dir = '../dataset/brain_train/brain_train'
brain_dataset_seg_dir = '/media/lab426/My Passport1/MOOD/brain/img'
abdom_dataset_origin_dir = '../dataset/abdom_train/abdom_train'
abdom_dataset_seg_dir = '/media/lab426/My Passport1/MOOD/abdom/img'

if train_dataset_id == 0:
    dataset_origin_dir = brain_dataset_origin_dir
    dataset_seg_dir = brain_dataset_seg_dir
    dataset_name = 'brain'
elif train_dataset_id == 1:
    dataset_origin_dir = abdom_dataset_origin_dir
    dataset_seg_dir = abdom_dataset_seg_dir
    dataset_name = 'abdom'
input_channel = 1
output_channel = 1

fold_num = 4
batch_size = 2
epoch_num = 50
num_workers = 8
learning_rate = 0.0001
resume = 0

entropy_loss_weight = 0.0005
sparse_shrink_thres = 0.0001
mem_dim_in = 4096

############ model saving dir path
local_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
# saving_root = '/media/lab426/My Passport1/MOOD/checkpoints'
saving_root = '/home/lab426/mfr/MOOD/checkpoints/'
model_setting = 'final_pretraining_{}_memdim_{}_thres_{}_lr_{}'.format(dataset_name, mem_dim_in, sparse_shrink_thres, learning_rate)
saving_model_path = os.path.join(saving_root, local_time + '_model_' + model_setting + '/')
utils.mkdir(saving_model_path)

blog_name = "{}_{}.csv".format(local_time, model_setting)
blog = open(os.path.join(saving_model_path, blog_name), "w")
blog_title = 'epoch,mse,entropy'
blog.write(blog_title + '\n')

###### data
data_num = len(os.listdir(dataset_seg_dir))
id_all = list(range(data_num))

train_dataset = dataset.PretrainingDataset_segmentation(dataset_origin_dir=dataset_origin_dir,
                                                        ids=id_all)
print(len(train_dataset))
train_data_loader = DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers,)

# load pretraining parameters
recon_model = AutoEncoderCov3DMem(chnum_in=input_channel, mem_dim=mem_dim_in, shrink_thres=sparse_shrink_thres)
recon_model_paths = [
'/home/lab426/mfr/MOOD/checkpoints/20220817_174944_model_final_pretraining_brain_memdim_4096_thres_0.0001_lr_0.0001/epoch_11.pt'
]
models = [torch.load(model_path) for model_path in recon_model_paths]
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


#########
device = torch.device("cuda")
recon_model.to(device)
recon_loss_func = nn.MSELoss().to(device)
entropy_loss_func = EntropyLossEncap().to(device)
train_optimizer = torch.optim.Adam(recon_model.parameters(), lr=learning_rate)

for epoch_idx in range(resume, epoch_num):
    for batch_idx, data in enumerate(train_data_loader):
        img_origin = data['img_origin'].to(device)

        recon_res = recon_model(img_origin)
        recon_img = recon_res['output']
        att_w = recon_res['att']

        loss_recon = recon_loss_func(recon_img, img_origin)
        entropy_loss = entropy_loss_func(att_w)
        loss = loss_recon + entropy_loss * entropy_loss_weight
        # ——————————————————————————————————————————————————
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        blog_txt = '{},{:.6f},{:.6f}'.format(epoch_idx, loss_recon.item(), entropy_loss.item())
        print(blog_txt)
        blog.write(blog_txt + '\n')

        # if random.random() < 0.02:
        #     batch_size_ = img_origin.shape[0]
        #     for i in range(batch_size_):
        #         recon_img_i = recon_img[i]
        #         recon_img_i = recon_img_i.float().squeeze().cpu().detach().numpy()
        #         output_dir = os.path.join(recon_dir, str(epoch_idx))
        #         if not os.path.exists(output_dir):
        #             os.makedirs(output_dir)
        #         output_path = os.path.join(output_dir, data['img_origin_name'][i].replace('.nii', '_train.nii'))
        #         nib.Nifti1Image(recon_img_i, affine=data['affine'][i]).to_filename(output_path)

    torch.save(recon_model.state_dict(),
               os.path.join(saving_model_path, 'epoch_{}.pt'.format(epoch_idx)))
