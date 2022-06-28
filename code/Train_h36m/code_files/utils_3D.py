import numpy as np
import torch

import numpy as np
import torch

# from utils.utils import wrap
# from utils.quaternion import qrot, qinverse


# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape)-1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape)-1)
    return (v + 2 * (q[..., :1] * uv + uuv))
    
    
def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape)-1)

import torch
import torch
import numpy as np
import hashlib
from torch.utils.data import DataLoader
# from utils.metrics import *

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result

def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value
    
# Evaluation

def eval(net, dataset, batch_size=64, max_batch= 10000, mean_zero= False, shuffle= False, useEncoder=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = Dataloader(dataset, batch_size=batch_size, shuffle= shuffle)

    mpjpe_record = []
    for it, batch in enumerate(data_loader):
        
        if it+1 > max_batch:
            break
        
        with torch.no_grad():
            b_size = batch['position_2d'].shape[0]
        if useEncoder:
            output, mu, log_var = net.forward(torch.tensor(batch['position_3d'], device=device), mean_zero=mean_zero)
        else:
            output = net.sample(b_size, torch.tensor(batch['position_2d'], device=device), mean_zero= mean_zero)
        
    output = torch.reshape(output, (batch))

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape)-1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape)-1)
    return (v + 2 * (q[..., :1] * uv + uuv))
    
    
def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape)-1)











def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]

    
def image_coordinates(X, w, h):
    assert X.shape[-1] == 2
    
    # Reverse camera frame normalization
    return (X + [1, h/w])*w/2
    

def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) # Rotate and translate

    
def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

    
def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    # print("------")
    # print(camera_params)
    # print(camera_params[..., 4:7])
    # print(XX.shape)
    # print(k.shape)
    # print(torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1).shape)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    XXX = XX*(radial + tan) + p*r2
    
    return f*XXX + c

def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    
    return f*XX + c

# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np

def absolute_to_root_relative(joints, root_index):
    root = joints.narrow(-2, root_index, 1)
    return joints - root

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance), #protocol #1
    """
    assert predicted.shape == target.shape
    predicted = absolute_to_root_relative(predicted, 0)
    target = absolute_to_root_relative(target, 0)

    return torch.mean(torch.norm(predicted-target, dim=len(predicted.shape)-1))

def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    print(target.shape)
    print(target)
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))


# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# skeleton.py-> Done

import numpy as np

class Skeleton:
    def __init__(self, parents, joints_left, joints_right): 
        assert len(joints_left) == len(joints_right)

        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata() # 1005?

    def num_joints(self):
        return len(self._parents)

    def parents(self):
        return self._has_children

    def children(self):
        return self._children


    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove'.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]
                
        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)
        
        
        if self._joints_left is not None:
            new_joints_left = []
            for joint in self._joints_left:
                if joint in valid_joints:
                    new_joints_left.append(joint - index_offsets[joint])
            self._joints_left = new_joints_left
        if self._joints_right is not None:
            new_joints_right = []
            for joint in self._joints_right:
                if joint in valid_joints:
                    new_joints_right.append(joint - index_offsets[joint])
            self._joints_right = new_joints_right

        self._compute_metadata()
        
        return valid_joints
    
    def joints_left(self):
        return self._joints_left
    
    def joints_right(self):
        return self._joints_right
        
    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)


import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
from torch.utils.data import DataLoader
#from utils.camera import *
import torch


def visualization(gt_2d, skeleton, keypoints_metadata):
    parents = skeleton.parents
    lines = []
    size=6
    fig = plt.figure(figsize=(size*2, size))
    ax_input = fig.add_subplot(1, 1, 1)
    ax_input.set_aspect('equal')
    ax_input.set(xlim=(0, 1000), ylim=(0, 1000))
    ax_input.set_title('Input')
    ax_input.set_axis_off()

    ax = fig.add_subplot(1, 2, 2)
    # ax.view_init(elev=15., azim=azim)
    # ax.set_xlim3d([-radius/2, radius/2])
    # ax.set_zlim3d([0, radius])
    # ax.set_ylim3d([-radius/2, radius/2])

    joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
    colors_2d = np.full(gt_2d.shape[0], 'black')
    colors_2d[joints_right_2d] = 'red'

    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue
        if keypoints_metadata['layout_name'] != 'coco':
            # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
            lines.append(ax_input.plot([gt_2d[j, 0], gt_2d[j_parent, 0]],
                                    [gt_2d[j, 1], gt_2d[j_parent, 1]], color='pink'))

            col = 'red' if j in skeleton.joints_right() else 'black'

    ax_input.scatter(*gt_2d.T, 10, color=colors_2d, edgecolors='white', zorder=10)
    ax_input.invert_yaxis()
    
    plt.show()

def visualization3d_no_gt(gt_2d, pred_3d, skeleton, keypoints_metadata, azim):
    parents = skeleton.parents()
    lines = []
    size=6
    fig = plt.figure(figsize=(size*2, size))
    ax_input = fig.add_subplot(1, 2, 1)
    ax_input.set_aspect('equal')
    ax_input.set(xlim=(0, 1000), ylim=(0, 1000))
    ax_input.set_title('Input')
    # ax_input.set_axis_off()

    radius = 1.7
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.view_init(elev=15., azim=azim)
    ax.set_xlim3d([-radius/2, radius/2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius/2, radius/2])
    ax.set_title('3D Prediction')
    try:
        ax.set_aspect('equal')
    except NotImplementedError:
        ax.set_aspect('auto')

    joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]

    input_colors_2d = np.full(gt_2d.shape[0], 'lightgray')
    input_colors_2d[joints_right_2d] = 'pink'

    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue
        if keypoints_metadata['layout_name'] != 'coco':
            # 2d_gt
            ax_input.plot([gt_2d[j, 0], gt_2d[j_parent, 0]],
                                    [gt_2d[j, 1], gt_2d[j_parent, 1]], color='pink')

            col_pred = 'red' if j in skeleton.joints_right() else 'black'

            ax.plot([pred_3d[j, 0], pred_3d[j_parent, 0]],
                    [pred_3d[j, 1], pred_3d[j_parent, 1]],
                    [pred_3d[j, 2], pred_3d[j_parent, 2]], zdir='z', c=col_pred)
                    

    ax_input.scatter(*gt_2d.T, 10, color=input_colors_2d, edgecolors='white', zorder=10)
    # ax_input.invert_yaxis()
    
    plt.savefig("test.jpg")
    plt.close()

def visualization3d(condition_2d, gt_2d, gt_3d, pred_3d, skeleton, keypoints_metadata, azim, path):
    parents = skeleton.parents()
    lines = []
    size=6
    fig = plt.figure(figsize=(size*2, size))
    ax_input = fig.add_subplot(1, 2, 1)
    ax_input.set_aspect('equal')
    ax_input.set(xlim=(0, 1000), ylim=(0, 1000))
    ax_input.set_title('Input')
    # ax_input.set_axis_off()

    radius = 1.7
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.view_init(elev=15., azim=azim)
    ax.set_xlim3d([-radius/2, radius/2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius/2, radius/2])
    ax.set_title('3D Prediction')
    try:
        ax.set_aspect('equal')
    except NotImplementedError:
        ax.set_aspect('auto')

    joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]

    condition_colors_2d = np.full(condition_2d.shape[0], 'black')
    condition_colors_2d[joints_right_2d] = 'red'

    input_colors_2d = np.full(gt_2d.shape[0], 'lightgray')
    input_colors_2d[joints_right_2d] = 'pink'

    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue
        if keypoints_metadata['layout_name'] != 'coco':
            # 2d_gt
            ax_input.plot([gt_2d[j, 0], gt_2d[j_parent, 0]],
                                    [gt_2d[j, 1], gt_2d[j_parent, 1]], color='pink')

            # 2d_input
            ax_input.plot([condition_2d[j, 0], condition_2d[j_parent, 0]],
                                    [condition_2d[j, 1], condition_2d[j_parent, 1]], color='red')

            col_pred = 'red' if j in skeleton.joints_right() else 'black'
            col_gt = 'green' if j in skeleton.joints_right() else 'blue'
            ax.plot([pred_3d[j, 0], pred_3d[j_parent, 0]],
                    [pred_3d[j, 1], pred_3d[j_parent, 1]],
                    [pred_3d[j, 2], pred_3d[j_parent, 2]], zdir='z', c=col_pred)
                    
            ax.plot([gt_3d[j, 0], gt_3d[j_parent, 0]],
                [gt_3d[j, 1], gt_3d[j_parent, 1]],
                [gt_3d[j, 2], gt_3d[j_parent, 2]], zdir='z', c=col_gt)

    ax_input.scatter(*gt_2d.T, 10, color=input_colors_2d, edgecolors='white', zorder=10)
    ax_input.scatter(*condition_2d.T, 10, color=condition_colors_2d, edgecolors='black', zorder=10)
    ax_input.invert_yaxis()
    
    plt.savefig(path)
    plt.close()

def draw_result(net, dataset, num_batch, fname, train = True, sample_num=4, mean_zero=False, shuffle=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader= DataLoader(dataset=dataset,
               batch_size=4,
               shuffle=shuffle,
               num_workers=0,
               pin_memory=True)

    for it, batch in enumerate(data_loader):
        if it == num_batch:
            break
        batch_size = batch['position_3d'].shape[0]

        intrinsic = [cam['intrinsic'] for cam in batch['camera']]
        intrinsic = torch.stack(intrinsic)

        for i in range(0, batch_size):
            skeleton = dataset.skeleton()
            keypoints_metadata = dataset.keypoints_metadata()

            ca = dataset.cameras()[batch['subject'][i]][batch['viz_camera'][i]]
            res_w = ca['res_w']
            res_h = ca['res_h']
            azim = ca['azimuth']

            with torch.no_grad():
                gt_2d = project_to_2d(batch['position_3d'], intrinsic[i])
                c = torch.tensor(batch['position_2d'][i] , device=device)  # CPN 2D pose prediction
                c = c.repeat(sample_num, 1)
                c = torch.reshape(c, (sample_num, 17, 2))
                output = net.sample(sample_num, c, mean_zero = mean_zero)
                cur_gt_2d = image_coordinates(gt_2d[i].numpy(), w=res_w, h=res_h)
                cur_cpm = image_coordinates(c[i].cpu().numpy(), w=res_w, h=res_h)
                for sample_idx in range(sample_num):
                    sample_output = camera_to_world(output[sample_idx].cpu(), ca['orientation'],ca['translation'])
                    if train:   
                        name = fname+ "batch{}_sample_num{}_train.png".format(i, sample_idx)
                    else:
                        name = fname+ "batch{}_sample_num{}_test.png".format(i, sample_idx)
                    visualization3d(cur_cpm, cur_gt_2d, sample_output, batch['position_3d_world'][i], skeleton, keypoints_metadata, azim, name)


def draw_result_reproject(net, dataset, num_batch, fname, train=True, sample_num=4, mean_zero=False, use_mean=False, shuffle=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader= DataLoader(dataset=dataset,
               batch_size=sample_num,
               shuffle=shuffle,
               num_workers=0,
               pin_memory=True)

    for it, batch in enumerate(data_loader):
        if it == num_batch:
            break
        batch_size = batch['position_3d'].shape[0]

        intrinsic = [cam['intrinsic'] for cam in batch['camera']]
        intrinsic = torch.stack(intrinsic)

        for i in range(0, batch_size):
            skeleton = dataset.skeleton()
            keypoints_metadata = dataset.keypoints_metadata()

            ca = dataset.cameras()[batch['subject'][i]][batch['viz_camera'][i]]
            res_w = ca['res_w']
            res_h = ca['res_h']
            azim = ca['azimuth']

            with torch.no_grad():
                gt_2d = project_to_2d(batch['position_3d'], intrinsic[i])
                c = torch.tensor(batch['position_2d'][i] , device=device)  # CPN 2D pose prediction
                c = c.repeat(sample_num, 1)
                c = torch.reshape(c, (sample_num, 17, 2))
                output = net.sample(sample_num, c, mean_zero = mean_zero)
                cur_gt_2d = image_coordinates(gt_2d[i].numpy(), w=res_w, h=res_h)
                reproject_2d = project_to_2d(output.cpu(), intrinsic[i])
                reproject_2d = image_coordinates(reproject_2d.numpy(), w=res_w, h=res_h)

                # cur_cpm = image_coordinates(c[i].cpu().numpy(), w=res_w, h=res_h)
                for sample_idx in range(sample_num):
                    sample_output = camera_to_world(output[sample_idx].cpu(), ca['orientation'],ca['translation'])
                    if train:
                        name = fname+ "batch{}_sample_num{}_train.png".format(i, sample_idx)
                    else:
                        name = fname+ "batch{}_sample_num{}_test.png".format(i, sample_idx)
                    visualization3d(reproject_2d[sample_idx], cur_gt_2d, sample_output, batch['position_3d_world'][i], skeleton, keypoints_metadata, azim, name)
