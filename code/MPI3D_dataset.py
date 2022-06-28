
#MPI3D_dataset
# load the function in MPI3D_utils folder

import re
import os
import sys
import os

import cv2 # load image
import pickle # create annotation file
from glob import iglob
from os import path

import h5py
import matplotlib.pyplot as plt
import numpy as np
# import torch.utils.data
import torch
from PIL import Image, ImageOps
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
from tqdm import tqdm
from pose3d_utils.coords import homogeneous_to_cartesian, ensure_homogeneous

# from MPI3D_utils.margipose.data import PoseDataset, collate
# from MPI3D_utils.margipose.data.mpi_inf_3dhp.common import Annotations, parse_camera_calibration, Constants, \
#     MpiInf3dhpSkeletonDesc
# from MPI3D_utils.margipose.data.skeleton import CanonicalSkeletonDesc, VNect_Common_Skeleton
# from MPI3D_utils.margipose.data_specs import DataSpecs, ImageSpecs, JointsSpecs
# from MPI3D_utils.margipose.eval import prepare_for_3d_evaluation, gather_3d_metrics
from pose_view_disentangle.data.mpi_inf_3dhp import MpiInf3dDataset

#from MPI3D_utils.pose_view_disentangle.data import PoseDataset, make_dataloader
from pose_view_disentangle.utils import randomRotate, seed_all, init_algorithms, timer, generator_timer

from pose_view_disentangle.data import PoseDataset, make_dataloader
# from MPI3D_utils.pose_view_disentangle.data.mixed import MixedPoseDataset
from pose_view_disentangle.data.get_dataset import get_dataset
# from MPI3D_utils.pose_view_disentangle.train_helpers import visualise_predictions, progress_iter

from pose_view_disentangle.data_specs import DataSpecs, ImageSpecs, JointsSpecs
from pose_view_disentangle.data.skeleton import CanonicalSkeletonDesc16, CanonicalSkeletonDesc
from utils_my import *
import matplotlib.pyplot as plt

# initial heatmap #https://github.com/SensorsINI/DHP19/blob/master/heatmap_generation_example.ipynb
img_h = 256
img_w = 256
n_joints = 17
heatmaps = np.zeros((img_h, img_w, n_joints))
hm_gauss = 2 # most used
print(heatmaps.shape)

# plt.figure()
# plt.imshow(img, cmap= 'gray')

def vis_try_ht_pred(info, index):
    # heatmap # info放target
    b =torch.zeros((64, 64))
    for i in range(15): 
        b=info[i]
        plt.imshow(b, cmap='magma')
        # plt.savefig("../1115_debug_vis_pred/Training_pred_heatmap_{}_{}.png".format(index, i))
        # plt.show()
    return print("Done with vis heatmap(Pred) on Training")

if __name__ == "__main__":
    print("calvin")#
    # data_dir = '/ssd/HumanPoseDataset/3D/mpi3d/train'
    data_dir = '/d/'
    data_specs = DataSpecs(
            ImageSpecs(256, mean=ImageSpecs.IMAGENET_MEAN, stddev=ImageSpecs.IMAGENET_STDDEV, heatmap_size=64, sigma=1),
            JointsSpecs(CanonicalSkeletonDesc, n_dims=3),
        )

    subset = 'train'
    print(data_dir)
    mpi_3d_dataset = MpiInf3dDataset(path.join(data_dir, 'mpi3d', subset),
                        data_specs=data_specs,
                        use_aug=(False and not subset.startswith('test')))

    # 0626 draw guassion heatmap
    # joints_gt = np.array([[ 9.9628e-03, -6.4930e-01, -2.4049e-02,  1.0000e+00],
    #     [ 1.9519e-02, -4.5496e-01, -1.3582e-02,  1.0000e+00],
    #     [ 3.0871e-02, -4.0089e-01, -1.2743e-01,  1.0000e+00],
    #     [ 4.6204e-02, -3.4980e-01, -4.4299e-01,  1.0000e+00],
    #     [ 8.3204e-02, -3.2111e-01, -7.3205e-01,  1.0000e+00],
    #     [ 4.5353e-03, -3.9070e-01,  9.1811e-02,  1.0000e+00],
    #     [-1.2910e-02, -3.8916e-01,  3.2760e-01,  1.0000e+00],
    #     [-1.0492e-02, -3.7640e-01,  4.8660e-01,  1.0000e+00],
    #     [ 4.1848e-02, -8.1222e-02, -1.0317e-01,  1.0000e+00],
    #     [ 8.9523e-03,  3.3346e-01, -3.7222e-02,  1.0000e+00],
    #     [-4.6287e-02,  6.4930e-01,  4.3990e-03,  1.0000e+00],
    #     [ 3.9011e-03, -1.0287e-01,  9.6889e-02,  1.0000e+00],
    #     [-1.3530e-02,  3.0776e-01,  7.3124e-02,  1.0000e+00],
    #     [-8.3204e-02,  6.3035e-01,  5.6353e-02,  1.0000e+00],
    #     [ 2.2279e-02, -9.2386e-02, -5.2592e-16,  1.0000e+00],
    #     [ 1.1862e-02, -2.6261e-01, -7.0161e-03,  1.0000e+00],
    #     [ 3.0286e-02, -5.1951e-01, -1.4636e-02,  1.0000e+00]])
    # # print(joints_gt.shape) (17, 4)
    # # 要確定要刪掉哪個joints -> 16
    # joints_2D = joints_gt[:, :2]
    # # print(joints_2D)
    # # print(joints_2D.shape) # (17, 2)


    # # make sure the 2D joints and its scatter
    # for i in range(16):
    #     plt.scatter(joints_2D[i][0], joints_2D[i][1], c ="blue")
    # plt.show()


    # exit()
    # heatmap_2D = np.zeros((16, 64, 64), dtype= np.float32) # 64*64 heatmap 設為0???

    # #self.num_joints = 16
    # # for i in range(self.num_joints)
    # hm_gauss = 2
    # for i in range(16):
    # # for i in range(joints_2D.shape[0]):

    #     heatmap_2D[i] = draw_gaussian(heatmap_2D[i], joints_2D[i], hm_gauss)

    
    # # vis_try_ht_pred(heatmap_2D, 1)

    # plt.imshow(heatmap_2D, cmap='hot', interpolation='nearest')
    # plt.show()
    # print("OK")
    # # 0626
    
    # 現在的問題: img跟joints_2D 
    # scale問題: guassion沒有map在0~64, 0~64區間
    i = 23000
    image = mpi_3d_dataset[i][0] # [3, 256, 256]
    image = image.permute(1, 2, 0).cpu().detach().numpy() # tensor to np.array and [256, 256, 3]
    print("image", image.shape)
    print("frame_info",mpi_3d_dataset[i][1]) 
    
    #joints_gt = mpi_3d_dataset[i][2]
    joints_gt = mpi_3d_dataset[i][1]['original_ske'] # add this one in dataloader -> delete later OR change the training function with _
    joints_2D = joints_gt[:, :2]
    image = (image * 255).astype(np.uint8) #https://github.com/slundberg/shap/issues/703
    plt.imshow(image, interpolation='nearest')
    # plt.show()
    # exit()
    # implot = plt.imshow(im)

    # plot the 2D joints (scatter)
    for i in range(16):
        plt.scatter(joints_2D[i][0], joints_2D[i][1], c ="blue")
    plt.show()

    exit()
    # 


    # read the iter data element
    for (input, _, target, index) in tqdm(mpi_3d_dataset):
        print(input)
        print
        print(target)
        print(_)

        print(index)

        exit()
        

    # mpi_3d_dataset = MpiInf3dDataset(data_dir=data_dir, data_specs=data_specs, use_aug=False)
    # print(type(mpi_3d_dataset)) # <class '__main__.MpiInf3dDataset'>
    print(len(mpi_3d_dataset))
    data_dict = mpi_3d_dataset.__getitem__(29500)
    

    """
    keys: dict_keys(['frame_ref', 'index', 'valid_depth', 'original_skel', 'camera_intrinsic', 'camera_extrinsic', 'target', 'transform_opts', 'joint_mask', 'input'])  
    
    """
    # dataset.__getitem__
    print(data_dict.keys())
    # print("frame_ref: ", data_dict['frame_ref'])
    # print("index", data_dict['index'])
    # print("input: ", data_dict['input'].shape) #'frame_ref', 'index', 'valid_depth', 'original_skel', 'camera_intrinsic', 'camera_extrinsic', 'target', 'transform_opts', 'joint_mask', 'input'
    # import torchvision.transforms as T
    # from PIL import Image

    # transform = T.ToPILImage()
    # img = transform(data_dict['input'])
    # img.show()
    # #img = data_dict['input'].permute(2, 1, 0)



    # print(data_dict['target'].shape)
    # print(data_dict['target'])

    
    
    # # print(data_dict['input'])
    exit()