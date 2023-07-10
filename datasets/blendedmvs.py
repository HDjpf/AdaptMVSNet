import os
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from utils import *

from datasets.data_io import *


def motion_blur(img: np.ndarray, max_kernel_size=3):
    # Either vertial, hozirontal or diagonal blur
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (max_kernel_size + 1) / 2) * 2 + 1  # make sure is odd
    center = int((ksize - 1) / 2)
    kernel = np.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid - center) + np.square(grid.T - center)) / (2. * var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    img = cv2.filter2D(img, -1, kernel)
    return img


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode="train", nviews=7, ndepths=128, interval_scale=1.06):
        super(MVSDataset, self).__init__()

        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.metas = self.build_list()
        self.transform = transforms.ColorJitter(brightness=0.25, contrast=(0.3, 1.5))
        self.stages = 4

    def build_list(self):
        metas = []
        proj_list = open(self.listfile).read().splitlines()

        for data_name in proj_list:
            dataset_folder = os.path.join(self.datapath, data_name)

            # read cluster
            cluster_path = os.path.join(dataset_folder, 'cams', 'pair.txt')
            cluster_lines = open(cluster_path).read().splitlines()
            image_num = int(cluster_lines[0])

            # get per-image info
            for idx in range(0, image_num):

                ref_id = int(cluster_lines[2 * idx + 1])
                cluster_info = cluster_lines[2 * idx + 2].rstrip().split()
                total_view_num = int(cluster_info[0])
                if total_view_num < self.nviews - 1:
                    continue

                src_ids = [int(x) for x in cluster_info[1::2]]

                metas.append((data_name, ref_id, src_ids))

        return metas

    def __len__(self):
        return len(self.metas)

    def read_img(self, filename):
        img = Image.open(filename)

        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        h, w, _ = np_img.shape
        np_img_ms = {
            "stage_3": cv2.resize(np_img, (w // 8, h // 8), interpolation=cv2.INTER_LINEAR),
            "stage_2": cv2.resize(np_img, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR),
            "stage_1": cv2.resize(np_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR),
            "stage_0": np_img
        }
        return np_img_ms

    def read_cam(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        #extrinsics[:3,3] = extrinsics[:3,3]*1000
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        depth_sample_num = float(lines[11].split()[2])
        depth_max = float(lines[11].split()[3])
        return intrinsics, extrinsics, depth_min, depth_max

    def read_mask(self, filename):
        masked_img = np.array(Image.open(filename), dtype=np.float32)
        mask = np.any(masked_img > 10, axis=2).astype(np.float32)

        h, w = mask.shape
        mask_ms = {
            "stage1": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": mask,
        }
        return mask_ms

    def read_depth_and_mask(self, filename, depth_min):
        # read pfm depth file
        # (576, 768)
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        mask = np.array(depth >= depth_min, dtype=np.float32)

        h, w, num = depth.shape
        mask_ms = {
            "stage_3": cv2.resize(mask, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
            "stage_2": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage_1": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage_0": cv2.resize(mask, (w // 1, h // 1), interpolation=cv2.INTER_NEAREST),
        }
        depth_ms = {
            "stage_3": cv2.resize(depth, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
            "stage_2": cv2.resize(depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage_1": cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage_0": cv2.resize(depth, (w // 1, h // 1), interpolation=cv2.INTER_NEAREST),
        }
        return depth_ms, mask_ms

    def K_inv(self,intrinsics):
        # intrinsics invers
        K = torch.eye(4, dtype=torch.float32)
        K[0, 0] = float(intrinsics[0, 0])/8
        K[1, 1] = float(intrinsics[1, 1])/8
        K[0, 2] = float(intrinsics[0, 2])/8
        K[1, 2] = float(intrinsics[1, 2])/8
        K_scaled = K.clone()
        stage_3 = np.linalg.inv(K_scaled)
        K[0, 0] = float(intrinsics[0, 0])/4
        K[1, 1] = float(intrinsics[1, 1])/4
        K[0, 2] = float(intrinsics[0, 2])/4
        K[1, 2] = float(intrinsics[1, 2])/4
        K_scaled = K.clone()
        stage_2 = np.linalg.inv(K_scaled)
        K[0, 0] = float(intrinsics[0, 0])/2
        K[1, 1] = float(intrinsics[1, 1])/2
        K[0, 2] = float(intrinsics[0, 2])/2
        K[1, 2] = float(intrinsics[1, 2])/2
        K_scaled = K.clone()
        stage_1 = np.linalg.inv(K_scaled)
        K[0, 0] = float(intrinsics[0, 0])
        K[1, 1] = float(intrinsics[1, 1])
        K[0, 2] = float(intrinsics[0, 2])
        K[1, 2] = float(intrinsics[1, 2])
        K_scaled = K.clone()
        stage_0 = np.linalg.inv(K_scaled)
        invK_scaled = {
            "stage_0": stage_0,
            "stage_1": stage_1,
            "stage_2": stage_2,
            "stage_3": stage_3,
        }
        return invK_scaled

    def __getitem__(self, idx):
        #idx = 15783
        data_name, ref_id, src_ids = self.metas[idx]
        view_ids = [ref_id] + src_ids[:self.nviews - 1]

        #data_name = '59f70ab1e5c5d366af29bf3e'

        imgs_0 = []
        imgs_1 = []
        imgs_2 = []
        imgs_3 = []
        img_paths = []
        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []
        mask = None
        depth = None
        depth_min = None
        depth_max = None
        invK_scaled = None

        for i, vid in enumerate(view_ids):

            img_path = os.path.join(self.datapath, data_name, 'blended_images', '%08d.jpg' % vid)
            cam_path = os.path.join(self.datapath, data_name, 'cams', '%08d_cam.txt' % vid)
            img_paths.append(img_path)

            imgs = self.read_img(img_path)
            imgs_0.append(imgs['stage_0'])
            imgs_1.append(imgs['stage_1'])
            imgs_2.append(imgs['stage_2'])
            imgs_3.append(imgs['stage_3'])

            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam(cam_path)
            # print(cam_path)

            # intrinsics invers
            invK_scaled = self.K_inv(intrinsics)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 0.125
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_3.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_2.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_1.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_0.append(proj_mat)

            if i == 0:
                depth_min = depth_min_
                depth_max = depth_max_
                ref_depth_path = os.path.join(self.datapath, data_name, 'rendered_depth_maps', '%08d.pfm' % vid)
                depth, mask = self.read_depth_and_mask(ref_depth_path, depth_min)
                for l in range(self.stages):
                    mask[f'stage_{l}'] = np.expand_dims(mask[f'stage_{l}'],2)
                    mask[f'stage_{l}'] = mask[f'stage_{l}'].transpose([2,0,1])
                    depth[f'stage_{l}'] = np.expand_dims(depth[f'stage_{l}'],2)
                    depth[f'stage_{l}'] = depth[f'stage_{l}'].transpose([2,0,1])

        # imgs: N*3*H0*W0, N is number of images
        imgs_0 = np.stack(imgs_0).transpose([0, 3, 1, 2])
        imgs_1 = np.stack(imgs_1).transpose([0, 3, 1, 2])
        imgs_2 = np.stack(imgs_2).transpose([0, 3, 1, 2])
        imgs_3 = np.stack(imgs_3).transpose([0, 3, 1, 2])

        imgs = {}
        imgs['stage_0'] = imgs_0
        imgs['stage_1'] = imgs_1
        imgs['stage_2'] = imgs_2
        imgs['stage_3'] = imgs_3

        # proj_matrices: N*4*4
        proj_matrices_0 = np.stack(proj_matrices_0)
        proj_matrices_1 = np.stack(proj_matrices_1)
        proj_matrices_2 = np.stack(proj_matrices_2)
        proj_matrices_3 = np.stack(proj_matrices_3)

        proj = {}
        proj['stage_3'] = proj_matrices_3
        proj['stage_2'] = proj_matrices_2
        proj['stage_1'] = proj_matrices_1
        proj['stage_0'] = proj_matrices_0

        # data is numpy array
        #print("depth_min:",depth_min,"    depth_max:",depth_max)
        return {"imgs": imgs,  # N*3*H0*W0
                    "proj_matrices": proj,  # N*4*4
                    "depth": depth,  # 1*H0 * W0
                    "depth_min": depth_min*1,  # scalar
                    "depth_max": depth_max*1,  # scalar
                    "invK_b44": invK_scaled,
                    "mask": mask,
                    "filename":data_name + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",
                    "idx":idx}  # 1*H0 * W0
