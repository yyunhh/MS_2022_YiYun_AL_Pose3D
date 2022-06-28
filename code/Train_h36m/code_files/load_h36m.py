import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import torch.utils.data as data
import numpy as np
import json
import cv2
import pickle
from utils_my import *
from debugger import *
from config_3D import ParseConfig
import logging
from opts import opts

def config():
    conf = ParseConfig()

    if conf.success:
        logging.info('Successfully loaded config')
    else:
        logging.warn('Could not load configuration! Exiting.')
        exit()

    return conf

def load_h36m(opt, split, precached_h36m=False):
    print('==> Loading initial 3D {} data.'.format(split))

    if precached_h36m:

      print("Loading h36m_cached_train")
      h36m_dict = np.load('h36m_cache_{}_15000.npy'.format(split), allow_pickle=True)
      h36m_dict_2 = np.load('h36m_cache_{}_15001.npy'.format(split), allow_pickle=True)
      h36m_dict_3 = np.load('h36m_cache_{}_15002.npy'.format(split), allow_pickle=True) #0224

      h36m_dict = h36m_dict[()]
      h36m_dict_2 = h36m_dict_2[()]
      h36m_dict_3 = h36m_dict_3[()] #0224
      print("Finish h36_cached_train")

      #h36m_dict = np.load(os.path.joint('h36m_cache_{}.npy'.format(split)), allow_pickle=True)
      return h36m_dict, h36m_dict_2, h36m_dict_3 #0224

    num_joints = 16
    mean_bone_length = 4000
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    aspect_ratio = 1.0 * opt.input_w / opt.input_h
    split = split
    opt = opt
    split_ = split[0].upper() + split[1:]
    image_path =  os.path.join(opt.data_dir, 'h36m', 'ECCV18_Challenge', split_, 'IMG') # change
    ann_path = os.path.join(opt.data_dir, 'h36m', 'msra_cache',
      'HM36_eccv_challenge_{}_cache'.format(split_),
      'HM36_eccv_challenge_{}_w288xh384_keypoint_jnt_bbox_db.pkl'.format(split_)
    )
    annot = pickle.load(open(ann_path, 'rb'))
    # dowmsample validation data
    idxs = np.arange(len(annot)) if split == 'train' \
                else np.arange(0, len(annot), 1 if opt.full_test else 10)
    num_samples = len(idxs)
    print('Loaded 3D {} {} samples'.format(split, num_samples))

    ind_to_jnt = {0: 'rfoot', 1: 'rknee', 2: 'rhip', 3: 'lhip', 4: 'lknee', 5: 'lfoot', 6: 'torso', 7: 'chest',
                    8: 'Neck', 9: 'Head', 10: 'rhand', 11: 'relb', 12: 'rsho', 13:'lsho', 14: 'lelb', 15:'lhand'}
    h36m_to_mpii = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]

    shuffle_ref_3d = [[3, 6], [2, 5], [1, 4], 
                          [16, 13], [15, 12], [14, 11]]

    shuffle_ref = [[0, 5], [1, 4], [2, 3], 
                        [10, 15], [11, 14], [12, 13]]

    h36m_dict = {'input': [], 'target': [], 'meta': [], 
            'reg_target': [], 'reg_ind': [], 'reg_mask': [], 'image':[], 'pick_index':[]}

    #for img_idx in tqdm(range(15001)): #num_samples
    #for img_idx in tqdm(range(15001, 30000)): #num_samples
    for img_idx in tqdm(range(30001, 35823)): #num_samples
    #for img_idx in tqdm(range(35823)): #num_samples
        try:
            path = '{}/{:05d}.jpg'.format(image_path, idxs[img_idx]+1)
            img = cv2.imread(path)
            
        except FileNotFoundError:
            logging.warning('Could not load filename: {}'.format(idxs[img_idx]+1))
            continue

        ann = annot[idxs[img_idx]]
        gt_3d = np.array(ann['joints_3d_relative'], np.float32)[:17]
        pts = np.array(ann['joints_3d'], np.float32)[h36m_to_mpii]
        c = np.array([ann['center_x'], ann['center_y']], dtype=np.float32)
        s = max(ann['width'], ann['height'])
        r = 0

        flipped = (split == 'train' and np.random.random() < opt.flip)
        if flipped:
            img = img[:, ::-1, :]
            
            c[0] = img.shape[1] - 1 - c[0]
            gt_3d[:, 0] *= -1
            pts[:, 0] = img.shape[1] - 1 - pts[:, 0]
            for e in shuffle_ref_3d:
                gt_3d[e[0]], gt_3d[e[1]] = gt_3d[e[1]].copy(), gt_3d[e[0]].copy()
            for e in shuffle_ref:
                pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

        s = min(s, max(img.shape[0], img.shape[1])) * 1.0
        s = np.array([s, s])

        s = adjust_aspect_ratio(s, aspect_ratio, opt.fit_short_side)

        trans_input = get_affine_transform(
        c, s, r, [opt.input_w, opt.input_h])
        inp = cv2.warpAffine(img, trans_input, (opt.input_w, opt.input_h),
                            flags=cv2.INTER_LINEAR)
        inp_copy = inp.copy()
        inp = (inp.astype(np.float32) / 256. - mean) / std
        inp = inp.transpose(2, 0, 1)

        trans_output = get_affine_transform(
        c, s, r, [opt.output_w, opt.output_h])
        out = np.zeros((num_joints, opt.output_h, opt.output_w), 
                        dtype=np.float32)
        reg_target = np.zeros((num_joints, 1), dtype=np.float32)
        reg_ind = np.zeros((num_joints), dtype=np.int64)
        reg_mask = np.zeros((num_joints), dtype=np.uint8)
        pts_crop = np.zeros((num_joints, 2), dtype=np.int32)
        for i in range(num_joints):
            pt = affine_transform(pts[i, :2], trans_output).astype(np.int32)
            if pt[0] >= 0 and pt[1] >=0 and pt[0] < opt.output_w \
                and pt[1] < opt.output_h:
                pts_crop[i] = pt
                out[i] = draw_gaussian(out[i], pt, opt.hm_gauss)
                reg_target[i] = pts[i, 2] / s[0] # assert not self.opt.fit_short_side
                reg_ind[i] = pt[1] * opt.output_w * num_joints + \
                            pt[0] * num_joints + i # note transposed
                
                reg_mask[i] = 1

        neck_x = pts[8][0]
        neck_y = pts[8][1]
        head_x = pts[9][0]
        head_y = pts[9][1]
        xy_1 = np.array([neck_x, neck_y], dtype=np.float32)
        xy_2 = np.array([head_x, head_y], dtype=np.float32)
        normalizer = np.linalg.norm(xy_1 - xy_2, ord=2)
        pick_index = img_idx
        #print('pick_index ', pick_index) #1224 把其關掉了不然會一直顯示pick index
        # print(xy_1, xy_2)
        # print(normalizer)
        
        meta = {'index' : idxs[img_idx], 'center' : c, 'scale' : s, 
                'gt_3d': gt_3d, 'pts_crop': pts_crop, 'normalizer':normalizer}

        h36m_dict['input'].append(inp)
        h36m_dict['target'].append(out)
        h36m_dict['meta'].append(meta)
        h36m_dict['reg_target'].append(reg_target)
        h36m_dict['reg_ind'].append(reg_ind)
        h36m_dict['reg_mask'].append(reg_mask)
        h36m_dict['image'].append(inp_copy)
        h36m_dict['pick_index'].append(pick_index)
    
    #之後改正確在打開

    np.save(file=os.path.join('h36m_cache_{}_15002.npy'.format(split)),
            arr=h36m_dict,
            allow_pickle=True)
  
    return h36m_dict
    


# def _get_part_info(self, index):
#     ann = self.annot[self.idxs[index]]
#     gt_3d = np.array(ann['joints_3d_relative'], np.float32)[:17]
#     pts = np.array(ann['joints_3d'], np.float32)[self.h36m_to_mpii]
#     # pts[:, :2] = np.array(ann['det_2d'], dtype=np.float32)[:, :2]
#     c = np.array([ann['center_x'], ann['center_y']], dtype=np.float32)
#     s = max(ann['width'], ann['height'])

#     return gt_3d, pts, c, s   

def __getitem__(self, index):
    if index == 0 and self.split == 'train':
        self.idxs = np.random.choice(self.num_samples, self.num_samples, replace=False)
    img = self._load_image(index)
    gt_3d, pts, c, s = self._get_part_info(index)
    r = 0
    
    if self.split == 'train':
      sf = self.opt.scale
      s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      # rf = self.opt.rotate
      # r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
      #    if np.random.random() <= 0.6 else 0

    flipped = (self.split == 'train' and np.random.random() < self.opt.flip)
    if flipped:
      img = img[:, ::-1, :]
      c[0] = img.shape[1] - 1 - c[0]
      gt_3d[:, 0] *= -1
      pts[:, 0] = img.shape[1] - 1 - pts[:, 0]
      for e in self.shuffle_ref_3d:
        gt_3d[e[0]], gt_3d[e[1]] = gt_3d[e[1]].copy(), gt_3d[e[0]].copy()
      for e in self.shuffle_ref:
        pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
    
    s = min(s, max(img.shape[0], img.shape[1])) * 1.0
    s = np.array([s, s])

    s = adjust_aspect_ratio(s, self.aspect_ratio, self.opt.fit_short_side)

    trans_input = get_affine_transform(
      c, s, r, [self.opt.input_w, self.opt.input_h])
    inp = cv2.warpAffine(img, trans_input, (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp_copy = inp.copy()
    inp = (inp.astype(np.float32) / 256. - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    trans_output = get_affine_transform(
      c, s, r, [self.opt.output_w, self.opt.output_h])
    out = np.zeros((self.num_joints, self.opt.output_h, self.opt.output_w), 
                    dtype=np.float32)
    reg_target = np.zeros((self.num_joints, 1), dtype=np.float32)
    reg_ind = np.zeros((self.num_joints), dtype=np.int64)
    reg_mask = np.zeros((self.num_joints), dtype=np.uint8)
    pts_crop = np.zeros((self.num_joints, 2), dtype=np.int32)
    for i in range(self.num_joints):
      pt = affine_transform(pts[i, :2], trans_output).astype(np.int32)
      if pt[0] >= 0 and pt[1] >=0 and pt[0] < self.opt.output_w \
        and pt[1] < self.opt.output_h:
        pts_crop[i] = pt
        out[i] = draw_gaussian(out[i], pt, self.opt.hm_gauss)
        reg_target[i] = pts[i, 2] / s[0] # assert not self.opt.fit_short_side
        reg_ind[i] = pt[1] * self.opt.output_w * self.num_joints + \
                     pt[0] * self.num_joints + i # note transposed
        
        reg_mask[i] = 1

    neck_x = pts[8][0]
    neck_y = pts[8][1]
    head_x = pts[9][0]
    head_y = pts[9][1]
    xy_1 = np.array([neck_x, neck_y], dtype=np.float32)
    xy_2 = np.array([head_x, head_y], dtype=np.float32)
    normalizer = np.linalg.norm(xy_1 - xy_2, ord=2)
    # print(xy_1, xy_2)
    # print(normalizer)
    

    meta = {'index' : self.idxs[index], 'center' : c, 'scale' : s, 
            'gt_3d': gt_3d, 'pts_crop': pts_crop, 'normalizer':normalizer}

    return {'input': inp, 'target': out, 'meta': meta, 
            'reg_target': reg_target, 'reg_ind': reg_ind, 'reg_mask': reg_mask, 'image':inp_copy}#, 'ht':out}

# def _load_image(self, index):

#     path = '{}/{:05d}.jpg'.format(self.image_path, self.idxs[index]+1)
#     #print(self.image_path)
#     img = cv2.imread(path)

#     return img

def main(opt,conf):
    print('Load info')
    args = conf.args # get config_3D之後要打開->用在2D部分
    args_3D = conf.args_3D # args_3D['loss_type'] # MSE
    load_h36m(opt, 'train')
    print("DONE")


if __name__ == '__main__':
  opt = opts().parse()
  conf = config() # get config_3D.py
  main(opt, conf) 

class H36M(data.Dataset):
  def __init__(self, opt, split, activelearning_obj):
    print('==> initializing 3D {} data.'.format(split))
    self.num_joints = 16
    self.num_eval_joints = 17
    self.h36m_to_mpii = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]
    self.mpii_to_h36m = [6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 9, \
                         13, 14, 15, 12, 11, 10]
    self.acc_idxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
    self.shuffle_ref = [[0, 5], [1, 4], [2, 3], 
                        [10, 15], [11, 14], [12, 13]]
    self.shuffle_ref_3d = [[3, 6], [2, 5], [1, 4], 
                          [16, 13], [15, 12], [14, 11]]
    self.edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
                  [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
                  [6, 8], [8, 9]]
    self.edges_3d = [[3, 2], [2, 1], [1, 0], [0, 4], [4, 5], [5, 6], \
                     [0, 7], [7, 8], [8, 10],\
                     [16, 15], [15, 14], [14, 8], [8, 11], [11, 12], [12, 13]]
    self.mean_bone_length = 4000
    self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    self.aspect_ratio = 1.0 * opt.input_w / opt.input_h
    self.split = split
    self.opt = opt
    split_ = split[0].upper() + split[1:]
    self.image_path =  os.path.join(
      self.opt.data_dir, 'h36m', 'ECCV18_Challenge', 'ECCV18_Challenge', split_, 'IMG') # change
    ann_path = os.path.join(
      self.opt.data_dir, 'h36m', 'msra_cache',
      'HM36_eccv_challenge_{}_cache'.format(split_),
      'HM36_eccv_challenge_{}_w288xh384_keypoint_jnt_bbox_db.pkl'.format(split_)
    )
    self.annot = pickle.load(open(ann_path, 'rb'))
    # dowmsample validation data
    self.idxs = np.arange(len(self.annot)) if split == 'train' \
                else np.arange(0, len(self.annot), 1 if opt.full_test else 10)
    self.num_samples = len(self.idxs)
    print('Loaded 3D {} {} samples'.format(split, self.num_samples))



  def _load_image(self, index):
    
    path = '{}/{:05d}.jpg'.format(self.image_path, self.idxs[index]+1)
    #print(self.image_path)
    img = cv2.imread(path)
    
    
    return img
  
  def _get_part_info(self, index):
    ann = self.annot[self.idxs[index]]
    gt_3d = np.array(ann['joints_3d_relative'], np.float32)[:17]
    pts = np.array(ann['joints_3d'], np.float32)[self.h36m_to_mpii]
    # pts[:, :2] = np.array(ann['det_2d'], dtype=np.float32)[:, :2]
    c = np.array([ann['center_x'], ann['center_y']], dtype=np.float32)
    s = max(ann['width'], ann['height'])

    return gt_3d, pts, c, s
    
      
  def __getitem__(self, index):
    if index == 0 and self.split == 'train':
        self.idxs = np.random.choice(self.num_samples, self.num_samples, replace=False)
    img = self._load_image(index)
    gt_3d, pts, c, s = self._get_part_info(index)
    r = 0
    
    if self.split == 'train':
      sf = self.opt.scale
      s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      # rf = self.opt.rotate
      # r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
      #    if np.random.random() <= 0.6 else 0

    flipped = (self.split == 'train' and np.random.random() < self.opt.flip)
    if flipped:
      img = img[:, ::-1, :]
      c[0] = img.shape[1] - 1 - c[0]
      gt_3d[:, 0] *= -1
      pts[:, 0] = img.shape[1] - 1 - pts[:, 0]
      for e in self.shuffle_ref_3d:
        gt_3d[e[0]], gt_3d[e[1]] = gt_3d[e[1]].copy(), gt_3d[e[0]].copy()
      for e in self.shuffle_ref:
        pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
    
    s = min(s, max(img.shape[0], img.shape[1])) * 1.0
    s = np.array([s, s])

    s = adjust_aspect_ratio(s, self.aspect_ratio, self.opt.fit_short_side)

    trans_input = get_affine_transform(
      c, s, r, [self.opt.input_w, self.opt.input_h])
    inp = cv2.warpAffine(img, trans_input, (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp_copy = inp.copy()
    inp = (inp.astype(np.float32) / 256. - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    trans_output = get_affine_transform(
      c, s, r, [self.opt.output_w, self.opt.output_h])
    out = np.zeros((self.num_joints, self.opt.output_h, self.opt.output_w), 
                    dtype=np.float32)
    reg_target = np.zeros((self.num_joints, 1), dtype=np.float32)
    reg_ind = np.zeros((self.num_joints), dtype=np.int64)
    reg_mask = np.zeros((self.num_joints), dtype=np.uint8)
    pts_crop = np.zeros((self.num_joints, 2), dtype=np.int32)
    for i in range(self.num_joints):
      pt = affine_transform(pts[i, :2], trans_output).astype(np.int32)
      if pt[0] >= 0 and pt[1] >=0 and pt[0] < self.opt.output_w \
        and pt[1] < self.opt.output_h:
        pts_crop[i] = pt
        out[i] = draw_gaussian(out[i], pt, self.opt.hm_gauss)
        reg_target[i] = pts[i, 2] / s[0] # assert not self.opt.fit_short_side
        reg_ind[i] = pt[1] * self.opt.output_w * self.num_joints + \
                     pt[0] * self.num_joints + i # note transposed
        
        reg_mask[i] = 1

    neck_x = pts[8][0]
    neck_y = pts[8][1]
    head_x = pts[9][0]
    head_y = pts[9][1]
    xy_1 = np.array([neck_x, neck_y], dtype=np.float32)
    xy_2 = np.array([head_x, head_y], dtype=np.float32)
    normalizer = np.linalg.norm(xy_1 - xy_2, ord=2)
    # print(xy_1, xy_2)
    # print(normalizer)
    

    meta = {'index' : self.idxs[index], 'center' : c, 'scale' : s, 
            'gt_3d': gt_3d, 'pts_crop': pts_crop, 'normalizer':normalizer}

    return {'input': inp, 'target': out, 'meta': meta, 
            'reg_target': reg_target, 'reg_ind': reg_ind, 'reg_mask': reg_mask, 'image':inp_copy}#, 'ht':out}
    
  def __len__(self):
    return self.num_samples


  def convert_eval_format(self, pred):
    pred_h36m = pred[self.mpii_to_h36m]
    pred_h36m[7] = (pred_h36m[0] + pred_h36m[8]) / 2
    pred_h36m[9] = (pred_h36m[8] + pred_h36m[10]) / 2
    sum_bone_length = self._get_bone_length(pred_h36m)
    mean_bone_length = self.mean_bone_length
    pred_h36m = pred_h36m * mean_bone_length / sum_bone_length
    return pred_h36m

  def _get_bone_length(self, pts):
    sum_bone_length = 0
    pts = np.concatenate([pts, (pts[14] + pts[11])[np.newaxis, :] / 2])
    for e in self.edges_3d:
      sum_bone_length += ((pts[e[0]] - pts[e[1]]) ** 2).sum() ** 0.5
    return sum_bone_length