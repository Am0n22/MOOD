import nibabel as nib
import random
import os
import numpy as np
import threading

'''
    本文档主要用于生成异常检测中自监督模型所使用的异常图像和对应的mask
    目前所采用的异常主要包含以下几种：
    1. 纯色区域
    2. 图像间像素迁移
    所使用的目标区域形状预计采用：
    1. 椭球
    2. 立方体
'''
def create_mask(origin_vol, anomaly_vol):
    shape_x, shape_y, shape_z = origin_vol.shape
    origin_area = np.where(origin_vol != 0, 1, 0)
    anomaly_area = np.where(anomaly_vol != 0, 1, 0)
    mask_area = origin_area * anomaly_area
    x_range = []
    y_range = []
    z_range = []
    for x_i in range(shape_x):
        if not np.max(mask_area[x_i, :, :]) == 0:
            x_range.append(x_i)
    for y_i in range(shape_y):
        if not np.max(mask_area[:, y_i, :]) == 0:
            y_range.append(y_i)
    for z_i in range(shape_z):
        if not np.max(mask_area[:, :, z_i]) == 0:
            z_range.append(z_i)
    print(x_range[0], x_range[-1], y_range[0], y_range[-1], z_range[0], z_range[-1])
    radius_x = int(random.uniform(0.1, 0.2) * (x_range[-1] - x_range[0]))
    radius_y = int(random.uniform(0.1, 0.2) * (y_range[-1] - y_range[0]))
    radius_z = int(random.uniform(0.1, 0.2) * (z_range[-1] - z_range[0]))
    center_x = random.randint(x_range[0]+2*radius_x, x_range[-1]-2*radius_x)
    center_y = random.randint(y_range[0]+2*radius_y, y_range[-1]-2*radius_y)
    center_z = random.randint(z_range[0]+2*radius_z, z_range[-1]-2*radius_z)
    mask = np.zeros(origin_vol.shape)
    for x in range(max(0, int(center_x - radius_x)), min(255, int(center_x + radius_x))):
        for y in range(max(0, int(center_y - radius_y)), min(255, int(center_y + radius_y))):
            for z in range(max(0, int(center_z - radius_z)), min(255, int(center_z + radius_z))):
                if (x - center_x) ** 2 / radius_x ** 2 + (y - center_y) ** 2 / radius_y ** 2 + (
                        z - center_z) ** 2 / radius_z ** 2 <= 1:
                    mask[x][y][z] = 1
    mask = mask * mask_area
    return mask

def mix_up(origin_vol, anomaly_vol, mask, mix_type='solid'):
    if mix_type == 'single':
        mixed_vol = origin_vol * (1 - mask) + random.random() * mask
    elif mix_type == 'merge':
        mixed_vol = origin_vol * (1 - mask) + anomaly_vol * mask * 1.5
    return mixed_vol

def random_flip(vol):
    random_x, random_y, random_z = [random.choice([1, -1]) for _ in range(3)]
    return vol[::random_x, ::random_y, ::random_z]

if __name__ == '__main__':
    dataset = 0
    brain_dataset_dir = '/home/lab426/mfr/MOOD/dataset/brain_train/brain_train'
    abdom_dataset_dir = '/home/lab426/mfr/MOOD/dataset/abdom_train/abdom_train'
    if dataset == 0:
        img_root = brain_dataset_dir
        new_root = '/media/lab426/My Passport1/MOOD/brain/img'
        mask_root = '/media/lab426/My Passport1/MOOD/brain/mask'
    else:
        img_root = abdom_dataset_dir
        new_root = '/media/lab426/My Passport1/MOOD/abdom/img'
        mask_root = '/media/lab426/My Passport1/MOOD/abdom/mask'
    # if not os.path.exists(new_root):
    #     os.makedirs(new_root)
    #     os.makedirs(mask_root)

    origin_list = os.listdir(img_root)
    print(len(origin_list))
    anomaly_list = os.listdir(img_root)
    random.shuffle(anomaly_list)

    def job(fold_id):
        print(fold_id)
        num = len(origin_list) / 10
        fold_list = origin_list[int(num*fold_id): int(num*(fold_id+1))]
        for i, origin_name in enumerate(fold_list):
            anomaly_name = anomaly_list[i]
            origin_vol = nib.load(os.path.join(img_root, origin_name))
            affine = origin_vol.affine
            origin_vol = np.asarray(origin_vol.get_fdata(), dtype='float32')
            anomaly_vol = nib.load(os.path.join(img_root, anomaly_name))
            anomaly_vol = np.asarray(anomaly_vol.get_fdata(), dtype='float32')
            if dataset == 1:
                origin_vol = origin_vol[::2, ::2, ::2]
                anomaly_vol = anomaly_vol[::2, ::2, ::2]

            mix_type = random.choice(['merge']*3 + ['single']*1)
            mask = create_mask(origin_vol, anomaly_vol)
            new_vol = mix_up(origin_vol, anomaly_vol, mask, mix_type=mix_type)
            new_img = nib.Nifti1Image(new_vol, affine)
            new_mask = nib.Nifti1Image(mask, affine)
            new_name = mix_type + '_' + origin_name[:-len('.nii.gz')] + '_' + anomaly_name
            nib.save(new_img, os.path.join(new_root, new_name))
            nib.save(new_mask, os.path.join(mask_root, new_name))
            print(new_name)

    threads = [None for i in range(10)]
    for i in range(10):
        threads[i] = threading.Thread(name='thread {}'.format(i), target=job, args=[i])
        threads[i].start()