"""
model 是用math
datalaoder 是用hg-py-3d
目前丟2D image進去跑
還沒寫inference
"""

import os
import copy
import logging
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLROnPlateau

from config_3D import ParseConfig
from utils import visualize_image
from utils import heatmap_loss
from activelearning import ActiveLearning
from load_h36m import load_h36m
from dataloader import load_hp_dataset
from dataloader import Dataset_MPII_LSPET_LSP
from evaluation import PercentageCorrectKeypoint
from models.learning_loss.LearningLoss import LearnLossActive
from models.stacked_hourglass.StackedHourglass import PoseNet as Hourglass
#from main import Train

# 3D favor
from models.stacked_hourglass.Simple import Encoder, Decoder,  weight_init
from models.stacked_hourglass.c_vae import cVAE
import datetime

import sys
#D:\Research\AL_Research\utils
# sys.path.insert(0, 'D:/Research/AL_Research/utils')
#from utils_3D import *
# from utils.utils import *
#from utilss.visualization import *
# from utils.camera import *
from dataloader_3D import * # #3D from Calvin
#from dataloader_my import * # my from hg-XX-3d

import os
import torch.utils.data as data
import numpy as np
import json
import cv2
import pickle
from utils_my import *
from debugger import *

def config():
    conf = ParseConfig()
    if conf.success:
        logging.info('Successfully loaded config')
    else:
        logging.warn('Could not load configuration! Exiting.')
        exit()

    return conf

ind_to_jnt = {0: 'rfoot', 1: 'rknee', 2: 'rhip', 3: 'lhip', 4: 'lknee', 5: 'lfoot', 6: 'torso', 7: 'chest',
                    8: 'Neck', 9: 'Head', 10: 'rhand', 11: 'relb', 12: 'rsho', 13:'lsho', 14: 'lelb', 15:'lhand'}

class H36M(data.Dataset):
  def __init__(self, opt, split):
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


    '''
    #可能要額外創一個class AL_H36M去挑要train的資料


    activelearning_samplers = {
    'random': activelearning_obj.random,
    'coreset': activelearning_obj.coreset_sampling,
    'learning_loss': activelearning_obj.learning_loss_sampling,
    'egl': activelearning_obj.expected_gradient_length_sampling,
    'entropy': activelearning_obj.multipeak_entropy
    }
    

    #self.train要有整個dataset的index
    #self.train = 

    # Dataset size
    self.dataset_size ={'h36m': {'{}'.format(split): self.num_samples}}
    #print(self.dataset_size) #{'h36m': {'train': 35832}}

    # self.index: 是要選的照片
    self.indices = activelearning_samplers[conf.active_learning_params['algorithm']](train = self.train, dataset_size=self.dataset_size)
    #activelearning_samplers 是ActiveLearning class 的object 會return annotation to indices
    # return 完的indices + 之前的 dataset info(dict) = train
    self.train = self.merge_dataset(datasets=[self.train_entire], indices=[self.indices])
    exit()
    '''

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

class H36M_AL(torch.utils.data.Dataset):
    '''
    inference pick indices:
    (1) self.indices, self.pick_indices
    (2) 
    '''
    def __init__(self, h36m_dict, h36m_dict_2, activelearning_obj, getitem_dump, conf, **kwargs):
        self.conf = conf
        self.hm_shape = kwargs['hourglass']['hm_shape']
        self.hm_peak = kwargs['misc']['hm_peak']
        self.threshold = kwargs['misc']['threshold'] * self.hm_peak
        self.model_save_path = getitem_dump

        self.h36m = h36m_dict
        self.h36m_2 = h36m_dict_2

        self.ind_to_jnt = list(ind_to_jnt.values())

        self.train_flag = False
        self.model_input_dataset = None

        activelearning_samplers = {
        'random': activelearning_obj.random,
        'coreset': activelearning_obj.coreset_sampling,
        'learning_loss': activelearning_obj.learning_loss_sampling,
        'entropy': activelearning_obj.multipeak_entropy}

        # Dataset sizes
        self.dataset_size ={'h36m': len(self.h36m['input'])}
        self.dataset_size_2 ={'h36m_2': len(self.h36m_2['input'])}

        print(self.dataset_size, '\t', self.dataset_size_2) 
    
        #其實這邊把原來dataset加上index就好，
        print(self.h36m.keys()) # dict_keys(['input', 'target', 'meta', 'reg_target', 'reg_ind', 'reg_mask', 'image'])
        print(self.h36m_2.keys())
        # self.indices = np.arange(self.dataset_size['h36m'], dtype=np.int64)

        #self.train_entire = np.concatenate((self.h36m, self.indices), axis=0)
        

        ##change merge two dict
        for k in self.h36m.keys():
            for i in self.h36m_2[k]:
                self.h36m[k].append(i)
        #merge
        self.train_entire = self.h36m
        print('check', len(self.train_entire['input']))

        # add 1229 to
        del self.h36m_2

        # print(self.train_entire['input'][32000])
        # print(self.train_entire['target'][32000])
        # print(self.train_entire['meta'][32000])
        # print(self.train_entire['image'][32000])
        
        #self.train_entire = self.h36m
        '''
        self.train_entire(dict): dict_keys(['input', 'target', 'meta', 'reg_target', 'reg_ind', 'reg_mask', 'image', 'pick_index'])
        print('train_entire:', type(self.train_entire)) 
        print('train_entire:', self.train_entire.keys())
        '''
        #self.train_entire = self.merge_dataset(datasets= [self.h36m], indices = [np.arange(self.dataset_size['h36m'], dtype=np.int64)])

        print("Pick the indices")
        self.indices, self.pick_indices = activelearning_samplers[conf.active_learning_params['algorithm']](
            train=self.train_entire, dataset_size=self.dataset_size) # AL.py return 兩個參數

        # self.indices = activelearning_samplers[conf.active_learning_params['algorithm']](
        #     train=self.train_entire, dataset_size=self.dataset_size)

        print("Turn into new dataset")
        self.train = self.merge_dataset(datasets=[self.train_entire], indices=[self.indices])
        print(len(self.train['pick_index']))

        # pick
        # print("Turn into new pick dataset")
        # self.pick = self.merge_dataset(datasets=[self.train_entire], indices =[self.pick_indices])
        # print('self.pick: ', type(self.pick), self.pick.keys())
        # print(len(self.pick['pick_index']))
        

        '''
        print('self.train: ', type(self.train))
        print(self.train.keys()) #dict_keys(['input', 'target', 'meta', 'reg_target', 'reg_ind', 'reg_mask', 'image', 'pick_index'])
        '''
        print('The dataset number to train: ', len(self.train['pick_index']))
        
        logging.info('\nFinal size of Training Data: {}'.format(len(self.train['input']))) #self.train['index'].shape[0]

        self.input_dataset(train=True)
        
    
    def __len__(self):
        #return self.model_input_dataset['input'].shape[0]
        return len(self.model_input_dataset['input'])

    def __getitem__(self, i):
        inp = self.model_input_dataset['input'][i]
        out = self.model_input_dataset['target'][i]
        meta = self.model_input_dataset['meta'][i]
        self.model_input_dataset['reg_target'][i]
        self.model_input_dataset['reg_ind'][i]
        self.model_input_dataset['reg_mask'][i]
        image = self.model_input_dataset['image'][i]

        # return {'input': inp, 'target': out, 'meta': meta, 'image':image}
        return inp, out, meta, image


    def input_dataset(self, train=False):
        if train:
            self.model_input_dataset = self.train
        return None

    def merge_dataset(self, datasets=None, indices=None):

        '''
        dataset(list)
        indices(list)

        datasets[0]: ['input', 'target', 'meta', 'reg_target', 'reg_ind', 'reg_mask', 'image']
        
        '''
        # print('dataset: ', type(dataset))
        # print('indices: ', type(indices), len(indices))

        merged_dataset = {}
        for key in datasets[0].keys():
            # print('indices', indices)
            # print('datasets: ', datasets)
            print("Do it", key)
            merged_dataset[key] = np.concatenate([np.array(data[key])[index_] for index_, data in zip(indices, datasets)], axis=0)

        merged_dataset['pick_index'] = np.arange(len(merged_dataset['input']))#.shape[0])

        print("Merge complete!", merged_dataset.keys())

        return merged_dataset

class H36M_pick(torch.utils.data.Dataset):
    def __init__(self, h36m_dict, conf, indices, **kwargs):
        self.conf =  conf 
        self.h36m = h36m_dict
        self.indices = indices
        self.ind_to_jnt = list(ind_to_jnt.values())
        self.model_input_dataset = None 

        self.train_entire = self.h36m

        self.pick = self.pick_dataset(datasets=[self.train_entire], indices=[self.indices])

        self.pick_input_dataset(train=True)

    def __len__(self):
        #return self.model_input_dataset['input'].shape[0]
        return len(self.model_input_dataset['input'])

    def __getitem__(self, i):
        inp = self.model_input_dataset['input'][i]
        out = self.model_input_dataset['target'][i]
        meta = self.model_input_dataset['meta'][i]
        self.model_input_dataset['reg_target'][i]
        self.model_input_dataset['reg_ind'][i]
        self.model_input_dataset['reg_mask'][i]
        image = self.model_input_dataset['image'][i]

        # return {'input': inp, 'target': out, 'meta': meta, 'image':image}
        return inp, out, meta, image


    def pick_dataset(self, datasets=None, indices=None):
        pick_dataset = {}
        for key in datasets[0].keys():
            pick_dataset[key] = np.concatenate([np.array(data[key])[index_] for index_, data in zip(indices, datasets)], axis=0)
        pick_dataset['pick_index'] = np.arange(len(pick_dataset['input']))
        print("Pick dataset complete!", pick_dataset.keys())

        return pick_dataset

    def pick_input_dataset(self, train=False):
        if train:
            self.model_input_dataset = self.pick
        return None


#################

# dataset_factory = {
#   'mpii': MPII,
#   'coco': COCO,
#   'fusion_3d': Fusion3D
# }

# task_factory = {
#   'human2d': (train, val), 
#   'human3d': (train_3d, val_3d)
# }


#### 1014 ####
'''
Ref: https://github.com/xingyizhou/pytorch-pose-hg-3d/blob/master/src/lib/train.py
input: 3D image and its 2D_gt
model: 2D model
'''
'''
from progress.bar import Bar
import time

def step(split, epoch, opt, data_loader, model, optimizer=None):
  if split == 'train':
    model.train()
  else:
    model.eval()
  
  crit = torch.nn.MSELoss()

  acc_idxs = data_loader.dataset.acc_idxs
  edges = data_loader.dataset.edges
  shuffle_ref = data_loader.dataset.shuffle_ref
  mean = data_loader.dataset.mean
  std = data_loader.dataset.std
  convert_eval_format = data_loader.dataset.convert_eval_format

  Loss, Acc = AverageMeter(), AverageMeter()
  data_time, batch_time = AverageMeter(), AverageMeter()
  preds = []
  
  nIters = len(data_loader)
  bar = Bar('{}'.format(opt.exp_id), max=nIters)
  
  end = time.time()
  for i, batch in enumerate(data_loader):
    data_time.update(time.time() - end)
    input, target, meta = batch['input'], batch['target'], batch['meta']
    input_var = input.cuda(device=opt.device, non_blocking=True)
    target_var = target.cuda(device=opt.device, non_blocking=True)

    output = model(input_var)

    loss = crit(output[-1]['hm'], target_var)
    for k in range(opt.num_stacks - 1):
      loss += crit(output[k], target_var)

    if split == 'train':
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    else:
      input_ = input.cpu().numpy().copy()
      input_[0] = flip(input_[0]).copy()[np.newaxis, ...]
      input_flip_var = torch.from_numpy(input_).cuda(
        device=opt.device, non_blocking=True)
      output_flip = model(input_flip_var)
      output_flip = shuffle_lr(
        flip(output_flip[-1]['hm'].detach().cpu().numpy()[0]), shuffle_ref)
      output_flip = output_flip.reshape(
        1, opt.num_output, opt.output_h, opt.output_w)
      # output_ = (output[-1].detach().cpu().numpy() + output_flip) / 2
      output_flip = torch.from_numpy(output_flip).cuda(
        device=opt.device, non_blocking=True)
      output[-1]['hm'] = (output[-1]['hm'] + output_flip) / 2
      print("OK")
      exit()
      pred, conf = get_preds(output[-1]['hm'].detach().cpu().numpy(), True)
      preds.append(convert_eval_format(pred, conf, meta)[0])
    
    Loss.update(loss.detach()[0], input.size(0))
    Acc.update(accuracy(output[-1]['hm'].detach().cpu().numpy(), 
                        target_var.detach().cpu().numpy(), acc_idxs))
   
    batch_time.update(time.time() - end)
    end = time.time()
    if not opt.hide_data_time:
      time_str = ' |Data {dt.avg:.3f}s({dt.val:.3f}s)' \
                 ' |Net {bt.avg:.3f}s'.format(dt = data_time,
                                                             bt = batch_time)
    else:
      time_str = ''
    Bar.suffix = '{split}: [{0}][{1}/{2}] |Total {total:} |ETA {eta:}' \
                 '|Loss {loss.avg:.5f} |Acc {Acc.avg:.4f}'\
                 '{time_str}'.format(epoch, i, nIters, total=bar.elapsed_td, 
                                     eta=bar.eta_td, loss=Loss, Acc=Acc, 
                                     split = split, time_str = time_str)
    if opt.print_iter > 0:
      if i % opt.print_iter == 0:
        print('{}| {}'.format(opt.exp_id, Bar.suffix))
    else:
      bar.next()
    if opt.debug >= 2:
      gt = get_preds(target.cpu().numpy()) * 4
      pred = get_preds(output[-1]['hm'].detach().cpu().numpy()) * 4
      debugger = Debugger(ipynb=opt.print_iter > 0, edges=edges)
      img = (input[0].numpy().transpose(1, 2, 0) * std + mean) * 256
      img = img.astype(np.uint8).copy()
      debugger.add_img(img)
      debugger.add_mask(
        cv2.resize(target[0].numpy().max(axis=0), 
                   (opt.input_w, opt.input_h)), img, 'target')
      debugger.add_mask(
        cv2.resize(output[-1]['hm'][0].detach().cpu().numpy().max(axis=0), 
                   (opt.input_w, opt.input_h)), img, 'pred')
      debugger.add_point_2d(pred[0], (255, 0, 0))
      debugger.add_point_2d(gt[0], (0, 0, 255))
      debugger.show_all_imgs(pause=True)

  bar.finish()
  return {'loss': Loss.avg, 
          'acc': Acc.avg, 
          'time': bar.elapsed_td.total_seconds() / 60.}, preds
  
def train(epoch, opt, train_loader, model, optimizer):
  return step('train', epoch, opt, train_loader, model, optimizer)
  
def val(epoch, opt, val_loader, model):
  return step('val', epoch, opt, val_loader, model)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

'''
#def load_models(conf=None, load_hg=True, load_learnloss=True, best_model=None, hg_param=None, model_dir=None):
def load_models(conf, load_hg, load_learnloss, best_model, hg_param, model_dir):   
    '''
    Initialize or load model(s): Hourglass, Learning Loss network
    :param conf: (Object of type ParseConfig) Contains the configuration for the experiment
    :param load_hg: (bool) Load Hourglass network
    :param load_learnloss: (bool) Load learning Loss network
    :param best_model: (bool) Load best model
    :param hg_param: (recheck type) Parameters for the Hourglass network
    :param model_dir: (string) Directory containing the model
    :return: (torch.nn x 2) Hourglass network, Learning Loss network
    '''
    print('Loading model')
    epoch = conf.model_load_epoch

    # Learn Loss model - Load or train from scratch, will be defined even if not needed
    if load_learnloss:
        logging.info('Loading Learning Loss model from: ' + model_dir)
        learnloss_ = LearnLossActive(conf.learning_loss_fc, conf.args['hourglass']['inp_dim'], 4,
                                     conf.learning_loss_original)

        if best_model:
            if conf.resume_training:
                path_ = '_hg'  # *_hg will be the Learning Loss model at the epoch where HG gave best results
            else:
                path_ = ''  # best Learning Loss model
            learnloss_.load_state_dict(torch.load(
                model_dir
                + '/model_checkpoints/best_ll_model_{}'.format(conf.learning_loss_obj) #best_model_learnloss_{}
                + path_
                + '.pth', map_location='cpu'))
        else:
            learnloss_.load_state_dict(torch.load(model_dir + 'model_checkpoints/best_ll_model_{}.pth'.format(epoch), map_location='cpu')) #model_epoch_{}_learnloss
    else:
        logging.info('Defining the Learning Loss module. Training from scratch!')
        learnloss_ = LearnLossActive(conf.learning_loss_fc, conf.args['hourglass']['inp_dim'], 4,
                                     conf.learning_loss_original)

    # Hourglass MODEL - Load or train from scratch
    if load_hg:
        # Load model
        logging.info('Loading Hourglass model from: ' + model_dir)
        net_ = Hourglass(nstack=hg_param['nstack'], inp_dim=hg_param['inp_dim'], oup_dim=hg_param['oup_dim'],
                         bn=hg_param['bn'], increase=hg_param['increase'])

        if best_model:
            net_.load_state_dict(torch.load(os.path.join(model_dir, 'model_checkpoints/best_model.pth'), map_location='cpu'))
        else:
            net_.load_state_dict(torch.load(os.path.join(model_dir, 'model_checkpoints/model_epoch_{}.pth'.format(epoch)), map_location='cpu'))

        logging.info("Successfully loaded Model")
        print("Successfully loaded Model")
        

    else:
        # Define model and train from scratch
        logging.info('Defining the network - Stacked Hourglass. Training from scratch!')
        net_ = Hourglass(nstack=hg_param['nstack'], inp_dim=hg_param['inp_dim'], oup_dim=hg_param['oup_dim'],
                         bn=hg_param['bn'], increase=hg_param['increase'])

    # Multi-GPU / Single GPU
    logging.info("Using " + str(torch.cuda.device_count()) + " GPUs")
    net = net_
    learnloss = learnloss_

    if torch.cuda.device_count() > 1:
        # Hourglass net has cuda definitions inside __init__(), specify for learnloss
        learnloss.cuda(torch.device('cuda:1'))
    else:
        # Hourglass net has cuda definitions inside __init__(), specify for learnloss
        learnloss.cuda(torch.device('cuda:0'))
    logging.info('Successful: Model transferred to GPUs.')

    return net, learnloss


# def load_models_3D(conf, load_3D, load_learnloss, best_model, model_dir):   
#     '''
#     Initialize or load model(s): 3D model, Learning Loss network
#     :param conf: (Object of type ParseConfig) Contains the configuration for the experiment
#     #param load_hg: (bool) Load Hourglass network
#     :param load_3D: (bool) Load 3D model
#     :param load_learnloss: (bool) Load learning Loss network
#     :param best_model: (bool) Load best model
#     :param hg_param: (recheck type) Parameters for the Hourglass network
#     :param model_dir: (string) Directory containing the model
#     :return: (torch.nn x 2) 3D model, Learning Loss network
#     '''
#     print('Loading 3D model')
#     epoch = conf.model_load_epoch

#     # Learn Loss model - Load or train from scratch, will be defined even if not needed
#     if load_learnloss:
#         logging.info('Loading Learning Loss model from: ' + model_dir)
#         learnloss_ = LearnLossActive(conf.learning_loss_fc, conf.args['hourglass']['inp_dim'], 4,
#                                      conf.learning_loss_original)

#         if best_model:
#             if conf.resume_training:
#                 path_ = '_hg'  # *_hg will be the Learning Loss model at the epoch where HG gave best results
#             else:
#                 path_ = ''  # best Learning Loss model
#             learnloss_.load_state_dict(torch.load(
#                 model_dir
#                 + 'model_checkpoints/best_model_learnloss_{}'.format(conf.learning_loss_obj)
#                 + path_
#                 + '.pth', map_location='cpu'))
#         else:
#             learnloss_.load_state_dict(torch.load(model_dir + 'model_checkpoints/model_epoch_{}_learnloss.pth'.format(epoch), map_location='cpu'))
#     else:
#         logging.info('Defining the Learning Loss module. Training from scratch!')
#         learnloss_ = LearnLossActive(conf.learning_loss_fc, conf.args['hourglass']['inp_dim'], 4,
#                                      conf.learning_loss_original)

#     # 3D MODEL - Load or train from scratch
#     if load_3D:
#         # Load model
#         logging.info('Loading 3D model from: ' + model_dir)
#         encoder = Encoder()
#         decoder = Decoder()
#         net_ = cVAE(encoder, decoder)

#         if best_model:
#             net_.load_state_dict(torch.load(os.path.join(model_dir, 'model_checkpoints/best_model.pth'), map_location='cpu'))
#         else:
#             net_.load_state_dict(torch.load(os.path.join(model_dir, 'model_checkpoints/model_epoch_{}.pth'.format(epoch)), map_location='cpu'))

#         logging.info("Successfully loaded Model")
#         print("Successfully loaded Model")
        

#     # else:
#     #     # Define model and train from scratch
#     #     logging.info('Defining the network - Stacked Hourglass. Training from scratch!')
#     #     net_ = Hourglass(nstack=hg_param['nstack'], inp_dim=hg_param['inp_dim'], oup_dim=hg_param['oup_dim'],
#     #                      bn=hg_param['bn'], increase=hg_param['increase'])

#     # Multi-GPU / Single GPU
#     logging.info("Using " + str(torch.cuda.device_count()) + " GPUs")
#     net = net_
#     learnloss = learnloss_

#     if torch.cuda.device_count() > 1:
#         # Hourglass net has cuda definitions inside __init__(), specify for learnloss
#         learnloss.cuda(torch.device('cuda:1'))
#     else:
#         # Hourglass net has cuda definitions inside __init__(), specify for learnloss
#         learnloss.cuda(torch.device('cuda:0'))
#     logging.info('Successful: Model transferred to GPUs.')

#     return net, learnloss

def define_hyperparams_3D(conf, net_3D, learnloss):
    print('start the hyperparameters 3D')
    hyperparameter_3D = dict()
    hyperparameter_3D['optimizer_config_3D'] = {'lr':conf.lr_3D}
    #hyperparameter_3D['loss_params'] = {'size_average': True} #先註解
    hyperparameter_3D['num_epochs_3D'] = conf.epochs_3D
    hyperparameter_3D['start_epoch'] = 0

    if conf.train_learning_loss:
        logging.info('Parameters of Learning Loss and Hourglass networks(3D) passed to Optimizer.')
        params_list = [{'params': net_3D.parameters()},
                        {'params': learnloss.parameters()}]

    else:
        logging.info('Parameters of Hourglass(3D) passed to Optimizer')
        params_list = [{'params': net_3D.parameters()}]

    hyperparameter_3D['optimizer'] = torch.optim.Adam(params_list, **hyperparameter_3D['optimizer_config_3D'])

    # if conf.resume_training:
    #     logging.info('Loading optimizer state dictionary')
    #     if conf.best_model:
    #         optim_dict = torch.load(conf.model_load_path + 'model_checkpoints/optim_best_model.tar')

    #     else:
    #         assert type(conf.model_load_epoch) == int, "Load epoch for optimizer not specified"
    #         optim_dict = torch.load(conf.model_load_path + 'model_checkpoints/optim_epoch_{}.tar'.format(
    #             conf.model_load_epoch))

    #     # If the previous experiment used learn_loss, ensure the llal model is loaded, with the correct optimizer
    #     assert optim_dict['learn_loss'] == conf.model_load_learnloss, "Learning Loss model needed to resume training"

    #     hyperparameters['optimizer'].load_state_dict(optim_dict['optimizer_load_state_dict'])
    #     logging.info('Optimizer state loaded successfully.\n')

    #     logging.info('Optimizer and Training parameters:\n')
    #     for key in optim_dict:
    #         if key == 'optimizer_load_state_dict':
    #             logging.info('Param group length: {}'.format(len(optim_dict[key]['param_groups'])))
    #         else:
    #             logging.info('Key: {}\tValue: {}'.format(key, optim_dict[key]))

    #     logging.info('\n')

    #     if conf.resume_training:
    #         hyperparameters['start_epoch'] = optim_dict['epoch']
    hyperparameter_3D['loss_fn'] = torch.nn.MSELoss(reduction='none')

    return hyperparameter_3D
    




def define_hyperparams(conf, net, learnloss):
    '''
    Defines the hyperparameters of the experiment
    :param conf: (Object of type ParseConfig) Contains the configuration for the experiment
    :param net: (torch.nn) HG model
    :return: (dict) hyperparameter dictionary
    '''
    logging.info('Initializing the hyperparameters for the experiment.')
    hyperparameters = dict()
    hyperparameters['optimizer_config'] = {
                                           'lr': conf.lr,
                                           'weight_decay': conf.weight_decay
                                          }
    hyperparameters['loss_params'] = {'size_average': True}
    hyperparameters['num_epochs'] = conf.epochs
    hyperparameters['start_epoch'] = 0  # Used for resume training

    # Parameters declared to the optimizer
    if conf.train_learning_loss:
        logging.info('Parameters of Learning Loss and Hourglass networks passed to Optimizer.')
        params_list = [{'params': net.parameters()},
                       {'params': learnloss.parameters()}]
    else:
        logging.info('Parameters of Hourglass passed to Optimizer')
        params_list = [{'params': net.parameters()}]

    hyperparameters['optimizer'] = torch.optim.Adam(params_list, **hyperparameters['optimizer_config'])

    if conf.resume_training:
        logging.info('Loading optimizer state dictionary')
        if conf.best_model:
            optim_dict = torch.load(conf.model_load_path + 'model_checkpoints/optim_best_model.tar')

        else:
            assert type(conf.model_load_epoch) == int, "Load epoch for optimizer not specified"
            optim_dict = torch.load(conf.model_load_path + 'model_checkpoints/optim_epoch_{}.tar'.format(
                conf.model_load_epoch))

        # If the previous experiment used learn_loss, ensure the llal model is loaded, with the correct optimizer
        assert optim_dict['learn_loss'] == conf.model_load_learnloss, "Learning Loss model needed to resume training"

        hyperparameters['optimizer'].load_state_dict(optim_dict['optimizer_load_state_dict'])
        logging.info('Optimizer state loaded successfully.\n')

        logging.info('Optimizer and Training parameters:\n')
        for key in optim_dict:
            if key == 'optimizer_load_state_dict':
                logging.info('Param group length: {}'.format(len(optim_dict[key]['param_groups'])))
            else:
                logging.info('Key: {}\tValue: {}'.format(key, optim_dict[key]))

        logging.info('\n')

        if conf.resume_training:
            hyperparameters['start_epoch'] = optim_dict['epoch']

    hyperparameters['loss_fn'] = torch.nn.MSELoss(reduction='none')

    return hyperparameters

class Train(object):
    def __init__(self, network, learnloss, hyperparameters, dataset_obj, conf, tb_writer, opt, val_dataloader):
        '''
        Class for training the model
        Training will train the Hourglass module
        :param network: (torch.nn) Hourglass model
        :param llal_ntwk: (torch.nn) Learning Loss model
        :param hyperparameters: (dict) Various hyperparameters used in training
        :param loc_object: (Object of LocalizationLoader) Controls the data fed into torch_dataloader
        :param model_save_path (string) The path directory where the training output will be logged.
        :param conf: (Object of ParseConfig) Contains the configurations for the model
        :param tb_writer: (Object of SummaryWriter) Tensorboard writer to log values
        :param wt_reg: (Bool) Whether to use weight regularization or not
        '''

        # Dataset Settings
        self.torch_dataloader = dataset_obj
        self.opt = opt
        self.val_dataloader = val_dataloader
        #self.dataset_obj = dataset_obj
        self.tb_writer = tb_writer                                           # Tensorboard writer
        self.network = network                                               # Hourglass network
        self.batch_size = conf.batch_size
        self.epoch = hyperparameters['num_epochs']
        self.hyperparameters = hyperparameters
        self.model_save_path = conf.model_save_path
        self.optimizer = hyperparameters['optimizer']                        # Adam / SGD
        self.loss_fn = hyperparameters['loss_fn']                            # MSE
        self.learning_rate = hyperparameters['optimizer_config']['lr']
        self.start_epoch = hyperparameters['start_epoch']                    # Used in case of resume training
        self.num_hm = conf.num_hm                                            # Number of heatmaps
        #self.joint_names = self.dataset_obj.ind_to_jnt
        self.hg_depth = 4                                                    # Depth of hourglass
        self.n_stack = conf.n_stack

        self.train_learning_loss = conf.train_learning_loss
        self.learnloss_network = learnloss
        self.learnloss_margin = conf.learning_loss_margin
        self.learnloss_warmup = conf.learning_loss_warmup
        self.learnloss_original = conf.learning_loss_original
        self.learnloss_obj = conf.learning_loss_obj

        # Stacked Hourglass scheduling
        if self.train_learning_loss:
            min_lr = [0.000003, conf.lr]
        else:
            min_lr = 0.000003

        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=8, cooldown=2, min_lr=min_lr, verbose=True)

        #self.torch_dataloader = torch.utils.data.DataLoader(self.dataset_obj, self.batch_size,
                                                            #shuffle=True, num_workers=8, drop_last=True)

        if torch.cuda.device_count() > 1:
            cuda_devices = [torch.device('cuda:0'), torch.device('cuda:1')]
        else:
            cuda_devices = [torch.device('cuda:0'), torch.device('cuda:0')]

        self.cuda_devices = cuda_devices
        if conf.learnloss_only:
            self.train_hg_bool = torch.tensor(0.0).cuda(cuda_devices[-1])
        else:
            self.train_hg_bool = torch.tensor(1.0).cuda(cuda_devices[-1])


    def train_model(self):
        '''
        Training Loop: Hourglass and/or Learning Loss
        :return: None
        '''

        print("Initializing training: Epochs - {}\tBatch Size - {}".format(self.hyperparameters['num_epochs'],
                                                                           self.batch_size))

        best_val_hg = np.inf
        best_val_learnloss = np.inf
        best_epoch_hg = -1
        best_epoch_learnloss = -1
        global_step = 0

        # Variable to store all the loss values for logging
        loss_across_epochs = []
        validation_across_epochs = []

        for e in range(self.start_epoch, self.epoch):
            epoch_loss = []
            epoch_loss_learnloss = []

            # Network alternates between train() and validate()
            self.network.train()
            # import gc
            # gc.collect()
            # torch.cuda.empty_cache()
            if self.train_learning_loss:
                self.learnloss_network.train()

            #self.dataset_obj.input_dataset(train=True)

            # Training loop
            logging.info('Training for epoch: {}'.format(e+1))
            #for (images, heatmaps, _, _, _, gt_per_image, split, _, _, _, joint_exist) in tqdm(self.torch_dataloader):
            for i, batch in enumerate(tqdm(self.torch_dataloader)):

                input, target, meta = batch['input'], batch['target'], batch['meta']
                #print("Get the input and its info")
                '''
                input: torch.Size([12, 3, 256, 256]) torch.float32
                target: torch.Size([12, 16, 64, 64]) torch.float32
                meta: (dict)
                '''                
                # vis_merge(input, meta, target, meta['index'])
                # exit()  
                input2 = input.permute(0, 2, 3, 1) 
                input2 = input2.cuda(device=opt.device, non_blocking=True)
                target = target.cuda(device=opt.device, non_blocking=True)

                
                for k in range(12):
                    self.vis_try_ht(target[k], meta['index'][k])
                
                exit()
                #input2 = input2.to(non_blocking=True, device= opt.device)#device=self.cuda_devices[-1])
                #target = target.to(non_blocking=True, device= opt.device)#device=self.cuda_devices[-1])

                outputs, hourglass_features = self.network(input2)
                #print(type(outputs)) #
                #print(outputs.dtype) # float32
                
                loss = heatmap_loss(outputs, target, self.n_stack)
                                
                '''
                o_c = outputs.cpu().detach().numpy()

                for k in range(12):
                    self.vis_try_ht_2(o_c[k], meta['index'][k])
                '''
                
                # Will clear the gradients of hourglass
                self.optimizer.zero_grad()
            
                
                learning_loss_ = loss.clone().detach()
                learning_loss_ = torch.mean(learning_loss_, dim=[1])

                loss = (torch.mean(loss)) * self.train_hg_bool
                self.tb_writer.add_scalar('Train/Loss_batch', torch.mean(loss), global_step)

                loss.backward()
                # Train the learning loss network
                if self.train_learning_loss:
                    loss_learnloss = self.learning_loss(hourglass_features, learning_loss_, self.learnloss_margin, input2, e)
                    loss_learnloss.backward()
                    epoch_loss_learnloss.append(loss_learnloss.cpu().data.numpy())

                # Weight update
                self.optimizer.step()
                global_step += 1

                # Store the loss per batch
                epoch_loss.append(loss.cpu().data.numpy())

            epoch_loss = np.mean(epoch_loss)
            if self.train_learning_loss:
                epoch_loss_learnloss = np.mean(epoch_loss_learnloss)

            # Returns average validation loss per element
            if self.train_learning_loss:
                validation_loss_hg, validation_learning_loss = self.validation(e)
            else:
                validation_loss_hg = self.validation(e)
                validation_learning_loss = 0.0

            # Learning rate scheduler on the HourGlass validation loss
            self.scheduler.step(validation_loss_hg)

            # TensorBoard Summaries
            self.tb_writer.add_scalar('Train', torch.Tensor([epoch_loss]), global_step)
            self.tb_writer.add_scalar('Validation/HG_Loss', torch.Tensor([validation_loss_hg]), global_step)
            if self.train_learning_loss:
                self.tb_writer.add_scalar('Validation/Learning_Loss', torch.Tensor([validation_learning_loss]), global_step)

            # Save the model
            torch.save(self.network.state_dict(),
                       self.model_save_path.format("model_epoch_{}.pth".format(e + 1)))

            if self.train_learning_loss:
                torch.save(self.learnloss_network.state_dict(),
                           self.model_save_path.format("model_epoch_{}_learnloss.pth".format(e + 1)))

            # For resume training ONLY:
            # If learn_loss, then optimizer will have two param groups
            # Hence during load, ensure llal module is loaded/not loaded

            torch.save({'epoch': e + 1,
            'optimizer_load_state_dict': self.optimizer.state_dict(),
            'mean_loss': epoch_loss,
            
            'learn_loss': self.train_learning_loss},
            self.model_save_path.format("optim_epoch_{}.tar".format(e + 1)))
            torch.save({'epoch': e + 1,
                        'optimizer_load_state_dict': self.optimizer.state_dict(),
                        'mean_loss': epoch_loss,
                        'mean_loss_validation': {'HG': validation_loss_hg, 'LearningLoss': validation_learning_loss},
                        'learn_loss': self.train_learning_loss},
                        self.model_save_path.format("optim_epoch_{}.tar".format(e + 1)))

            # Save if best model
            if best_val_hg > validation_loss_hg:
                torch.save(self.network.state_dict(),
                           self.model_save_path.format("best_model.pth"))

                torch.save(self.learnloss_network.state_dict(),
                           self.model_save_path.format("best_model_learnloss_hg.pth"))

                best_val_hg = validation_loss_hg
                best_epoch_hg = e + 1

                torch.save({'epoch': e + 1,
                            'optimizer_load_state_dict': self.optimizer.state_dict(),
                            'mean_loss_train': epoch_loss,
                            'mean_loss_validation': {'HG': validation_loss_hg, 'LearningLoss': validation_learning_loss},
                            'learn_loss': self.train_learning_loss},
                           self.model_save_path.format("optim_best_model.tar"))

            if self.train_learning_loss:
                if best_val_learnloss > validation_learning_loss and validation_learning_loss != 0.0:
                    torch.save(self.learnloss_network.state_dict(),
                               self.model_save_path.format("best_model_learnloss_{}.pth".format(self.learnloss_obj)))

                    best_val_learnloss = validation_learning_loss
                    best_epoch_learnloss = e + 1


            print("Loss at epoch {}/{}: (train) {}\t"
                  "Learning Loss: (train) {}\t"
                  "(validation: HG) {}\t"
                  "(Validation: Learning Loss) {}\t"
                  "(Best Model) {}".format(
                e+1,
                self.epoch,
                epoch_loss,
                epoch_loss_learnloss,
                validation_loss_hg,
                validation_learning_loss,
                best_epoch_hg))

            loss_across_epochs.append(epoch_loss)
            validation_across_epochs.append(validation_loss_hg)

            # Save the loss values
            f = open(self.model_save_path.format("loss_data.txt"), "w")
            f_ = open(self.model_save_path.format("validation_data.txt"), "w")
            f.write("\n".join([str(lsx) for lsx in loss_across_epochs]))
            f_.write("\n".join([str(lsx) for lsx in validation_across_epochs]))
            f.close()
            f_.close()

        self.tb_writer.close()
        logging.info("Model training completed\nBest validation loss (HG): {}\tBest Epoch: {}"
                     "\nBest validation loss (LLAL): {}\tBest Epoch: {}".format(
            best_val_hg, best_epoch_hg, best_val_learnloss, best_epoch_learnloss))

    def validation(self, e):
        '''
        Validation loss
        :param e: (int) Epoch
        :return: (Float): Mean validation loss per batch for Hourglass and Learning Loss (if LL activated in inc_config file.)
        '''
        with torch.no_grad():
            # Stores the loss for all batches
            epoch_val_hg = []

            if self.train_learning_loss:
                epoch_val_learnloss = []

            self.network.eval()
            if self.train_learning_loss:
                self.learnloss_network.eval()

            # Augmentation only needed in Training
            #self.dataset_obj.input_dataset(validate=True)

            # Compute and store batch-wise validation loss in a list
            logging.info('Validation for epoch: {}'.format(e+1))
            for i, batch in enumerate(tqdm(self.val_dataloader)):
                input_val, target_val, meta_val = batch['input'], batch['target'], batch['meta'] 

                input_val = input_val.to(non_blocking=True, device=self.cuda_devices[-1])

                input_val = input_val.permute(0, 2, 3, 1) 

                outputs_val, hourglass_features_val = self.network(input_val)

                target_val = target_val.to(non_blocking=True, device=self.cuda_devices[-1])

                loss_val_hg = heatmap_loss(outputs_val, target_val, self.n_stack)

                learning_loss_val = loss_val_hg.clone().detach()
                learning_loss_val = torch.mean(learning_loss_val, dim=[1])

                loss_val_hg = torch.mean(loss_val_hg)
                epoch_val_hg.append(loss_val_hg.cpu().data.numpy())

                if self.train_learning_loss:
                    loss_val_learnloss = self.learning_loss(hourglass_features_val, learning_loss_val, self.learnloss_margin, input_val, e)
                    epoch_val_learnloss.append(loss_val_learnloss.cpu().data.numpy())

            print("Validation Loss HG at epoch {}/{}: {}".format(e+1, self.epoch, np.mean(epoch_val_hg)))

            if self.train_learning_loss:
                print("Validation Learning Loss at epoch {}/{}: {}".format(e+1, self.epoch, np.mean(epoch_val_learnloss)))
                return np.mean(epoch_val_hg), np.mean(epoch_val_learnloss)

            else:
                return np.mean(epoch_val_hg)

    def learning_loss(self, hg_encodings, true_loss, margin, gt_per_img, epoch):
        '''
        Learning Loss module
        Refer:
        1. "Learning Loss For Active Learning, CVPR 2019"
        2. "A Mathematical Analysis of Learning Loss for Active Learning in Regression, CVPRW 2021"
        :param hg_encodings: (Dict of tensors) Intermediate (Hourglass) and penultimate layer output of the Hourglass network
        :param true_loss: (Tensor of shape [Batch Size]) Loss computed from HG prediction and ground truth
        :param margin: (scalar) tolerance margin between predicted losses
        :param gt_per_img: (Tensor, shape [Batch Size]) Number of ground truth per image
        :param epoch: (scalar) Epoch, used in learning loss warm start-up
        :return: (Torch scalar tensor) Learning Loss
        '''

        # Concatenate the layers instead of a linear combination
        with torch.no_grad():
            if self.learnloss_original:
                # hg_depth == 4 means depth is {1, 2, 3, 4}. If we want depth 5, range --> (1, 4+2)
                # encodings = torch.cat([hg_encodings[depth] for depth in range(1, self.hg_depth + 2)], dim=-1)
                encodings = hg_encodings['penultimate']

            else:
                # No longer concatenating, will now combine features through convolutional layers
                encodings = torch.cat([hg_encodings['feature_5'].reshape(self.batch_size, hg_encodings['feature_5'].shape[1], -1),               # 64 x 64
                                       hg_encodings['feature_4'].reshape(self.batch_size, hg_encodings['feature_4'].shape[1], -1),               # 32 x 32
                                       hg_encodings['feature_3'].reshape(self.batch_size, hg_encodings['feature_3'].shape[1], -1),               # 16 x 16
                                       hg_encodings['feature_2'].reshape(self.batch_size, hg_encodings['feature_2'].shape[1], -1),               # 8 x 8
                                       hg_encodings['feature_1'].reshape(self.batch_size, hg_encodings['feature_1'].shape[1], -1)], dim=2)       # 4 x 4

        emperical_loss, encodings = self.learnloss_network(encodings)
        emperical_loss = emperical_loss.squeeze()

        assert emperical_loss.shape == true_loss.shape, "Mismatch in Batch size for true and emperical loss"

        with torch.no_grad():
            # Scale the images as per the number of joints
            # To prevent DivideByZero. PyTorch does not throw an exception to DivideByZero
            gt_per_img = torch.sum(gt_per_img, dim=1)
            gt_per_img += 0.1
            if self.learnloss_obj == 'prob':
                true_loss = true_loss / gt_per_img

            # Splitting into pairs: (i, i+half)
            half_split = true_loss.shape[0] // 2

            true_loss_i = true_loss[: half_split]
            true_loss_j = true_loss[half_split: 2 * half_split]

        emp_loss_i = emperical_loss[: (emperical_loss.shape[0] // 2)]
        emp_loss_j = emperical_loss[(emperical_loss.shape[0] // 2): 2 * (emperical_loss.shape[0] // 2)]

        # Loss according to CVPR '19
        if self.learnloss_obj == 'pair':
            loss_sign = torch.sign(true_loss_i - true_loss_j)
            loss_emp = (emp_loss_i - emp_loss_j)

            # Learning Loss objective
            llal_loss = torch.max(torch.zeros(half_split, device=loss_sign.device), (-1 * (loss_sign * loss_emp)) + margin)

        # Loss according to CVPR '21
        elif self.learnloss_obj == 'prob':
            with torch.no_grad():
                true_loss_ = torch.cat([true_loss_i.reshape(-1, 1), true_loss_j.reshape(-1, 1)], dim=1)
                true_loss_scaled = true_loss_ / torch.sum(true_loss_, dim=1, keepdim=True)

            emp_loss_ = torch.cat([emp_loss_i.reshape(-1, 1), emp_loss_j.reshape(-1, 1)], dim=1)
            emp_loss_logsftmx = torch.nn.LogSoftmax(dim=1)(emp_loss_)
            llal_loss = torch.nn.KLDivLoss(reduction='batchmean')(input=emp_loss_logsftmx, target=true_loss_scaled)

        else:
            raise NotImplementedError('Currently only "pair" or "prob" supported. ')

        if self.learnloss_warmup <= epoch:
            return torch.mean(llal_loss)
        else:
            return 0.0 * torch.mean(llal_loss)

    def vis_merge(self, info, meta_info, target, file_name):
        '''
        info: (torch.Tensor)
        '''
        mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
        print('image index: ', meta_info['index'][0]+1)
        input2 = info[0].permute(1, 2, 0) 
        input2 = (input2*std + mean) # image 的matric

        input2_array = (input2.numpy()*255).astype(np.uint8)
        #print(input2_array.dtype)
        #cv2.imshow('image',input2_array)
        cv2.imwrite('../debug_vis/joints/image_img.png', input2_array) # RGB

        image_to_write = cv2.cvtColor(input2_array, cv2.COLOR_RGB2BGR) #cv2.COLOR_BGR2RGB
        cv2.imwrite('../debug_vis/joints/image_cvt.png', image_to_write) # RGB
        #cv2.imwrite('../debug_vis/joints/image.png', image_to_write)
        #cv2.waitKey(0)
    
        bottom_pic = input2_array
        #bottom_pic = cv2.resize(image_to_write, (64, 64), interpolation = cv2.INTER_AREA)

        b =torch.zeros((64, 64))
        for i in range(16): 
            b=target[0][i]
        
            b_array = (b.numpy()*255).astype(np.uint8)
            #print(b_array.dtype)
            heatmap = cv2.applyColorMap(b_array, cv2.COLORMAP_HOT)
            top_pic = cv2.resize(heatmap, (256, 256), interpolation = cv2.INTER_AREA)
            print(top_pic.shape)
            print(bottom_pic.shape)
            overlapping_pic = cv2.addWeighted(bottom_pic, 0.6, top_pic, 0.4, 0) #(256, 256, 3) and unit8
            cv2.imwrite('../debug_vis/joints/image_merge_joints_{}.png'.format(i), overlapping_pic)
            
            #plt.savefig("../debug_vis/joints/image_{}_joint_{}.png".format(file_name, i))

        return print("Merge!")

    def vis_try_ht(self, info, index):
        # heatmap # info放target
        b =torch.zeros((64, 64)).to('cuda:0')
        for i in range(15): 
            b+=info[i]
        b=b.to('cpu')
        plt.imshow(b, cmap='magma')
        plt.savefig("../1115_debug_vis/Training_GT_heatmap_{}.png".format(index))
        return print("Done with vis heatmap(GT) on Training")

    def vis_try_ht_2(self, info, index):
        # heatmap # info放target
        b =torch.zeros((64, 64))
        for i in range(15): 
            b+=info[0][i]
        plt.imshow(b, cmap='magma')
        plt.savefig("../debug_vis/Training_Pred_heatmap_{}.png".format(index))
        return print("Done with vis heatmap on Training")

    def vis_try(self, info, meta_info):
        #畫input(crop image)
        mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
        print('image index: ', meta_info['index'][0]+1)
        input2 = info[0].permute(1, 2, 0) 
        input2 = (input2*std + mean)
        plt.imshow(input2)
        plt.savefig("../debug_vis/Training_input_crop.png")
        
        return print("Done with vis input on Training")

class Train2(object):
    def __init__(self, network, learnloss, hyperparameters, dataset_obj, conf, tb_writer, opt, val_dataloader):
        '''
        Class for training the model
        Training will train the Hourglass module
        :param network: (torch.nn) Hourglass model
        :param llal_ntwk: (torch.nn) Learning Loss model
        :param hyperparameters: (dict) Various hyperparameters used in training
        :param loc_object: (Object of LocalizationLoader) Controls the data fed into torch_dataloader
        :param model_save_path (string) The path directory where the training output will be logged.
        :param conf: (Object of ParseConfig) Contains the configurations for the model
        :param tb_writer: (Object of SummaryWriter) Tensorboard writer to log values
        :param wt_reg: (Bool) Whether to use weight regularization or not
        '''

        # Dataset Settings
        self.torch_dataloader = dataset_obj
        self.opt = opt
        self.val_dataloader = val_dataloader
        #self.dataset_obj = dataset_obj
        self.tb_writer = tb_writer                                           # Tensorboard writer
        self.network = network                                               # Hourglass network
        self.batch_size = conf.batch_size
        self.epoch = hyperparameters['num_epochs']
        self.hyperparameters = hyperparameters
        self.model_save_path = conf.model_save_path
        self.optimizer = hyperparameters['optimizer']                        # Adam / SGD
        self.loss_fn = hyperparameters['loss_fn']                            # MSE
        self.learning_rate = hyperparameters['optimizer_config']['lr']
        self.start_epoch = hyperparameters['start_epoch']                    # Used in case of resume training
        self.num_hm = conf.num_hm                                            # Number of heatmaps
        #self.joint_names = self.dataset_obj.ind_to_jnt
        self.hg_depth = 4                                                    # Depth of hourglass
        self.n_stack = conf.n_stack

        self.train_learning_loss = conf.train_learning_loss
        self.learnloss_network = learnloss
        self.learnloss_margin = conf.learning_loss_margin
        self.learnloss_warmup = conf.learning_loss_warmup
        self.learnloss_original = conf.learning_loss_original
        self.learnloss_obj = conf.learning_loss_obj

        # Stacked Hourglass scheduling
        if self.train_learning_loss:
            min_lr = [0.000003, conf.lr]
        else:
            min_lr = 0.000003

        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=8, cooldown=2, min_lr=min_lr, verbose=True)

        #self.torch_dataloader = torch.utils.data.DataLoader(self.dataset_obj, self.batch_size,
                                                            #shuffle=True, num_workers=8, drop_last=True)

        if torch.cuda.device_count() > 1:
            cuda_devices = [torch.device('cuda:0'), torch.device('cuda:1')]
        else:
            cuda_devices = [torch.device('cuda:0'), torch.device('cuda:0')]

        self.cuda_devices = cuda_devices
        if conf.learnloss_only:
            self.train_hg_bool = torch.tensor(0.0).cuda(cuda_devices[-1])
        else:
            self.train_hg_bool = torch.tensor(1.0).cuda(cuda_devices[-1])


    def train_model(self):
        '''
        Training Loop: Hourglass and/or Learning Loss
        :return: None
        '''

        print("Initializing training: Epochs - {}\tBatch Size - {}".format(self.hyperparameters['num_epochs'],
                                                                           self.batch_size))

        best_val_hg = np.inf
        best_val_learnloss = np.inf
        best_epoch_hg = -1
        best_epoch_learnloss = -1
        global_step = 0

        # Variable to store all the loss values for logging
        loss_across_epochs = []
        validation_across_epochs = []

        for e in range(self.start_epoch, self.epoch):
            epoch_loss = []
            epoch_loss_learnloss = []

            # Network alternates between train() and validate()
            self.network.train()
            # import gc
            # gc.collect()
            # torch.cuda.empty_cache()
            if self.train_learning_loss:
                self.learnloss_network.train()

            #self.dataset_obj.input_dataset(train=True)

            # Training loop
            logging.info('Training for epoch: {}'.format(e+1))
            for (inp, out, meta, image) in tqdm(self.torch_dataloader):
                input2 = inp.permute(0, 2, 3, 1)
                input2 = input2.cuda(device=opt.device, non_blocking=True)
                target = out.cuda(device=opt.device, non_blocking=True)
            # for i, batch in enumerate(tqdm(self.torch_dataloader)):

            #     input, target, meta = batch['input'], batch['target'], batch['meta']
            #     #print("Get the input and its info")
            #     '''
            #     input: torch.Size([12, 3, 256, 256]) torch.float32
            #     target: torch.Size([12, 16, 64, 64]) torch.float32
            #     meta: (dict)
            #     '''                
            #     # vis_merge(input, meta, target, meta['index'])
            #     # exit()  
            #     input2 = input.permute(0, 2, 3, 1) 
            #     input2 = input2.cuda(device=opt.device, non_blocking=True)
            #     target = target.cuda(device=opt.device, non_blocking=True)
                
                #Open
                # for k in range(12):
                #     self.vis_try_ht(target[k], meta['index'][k])
                
                
                #input2 = input2.to(non_blocking=True, device= opt.device)#device=self.cuda_devices[-1])
                #target = target.to(non_blocking=True, device= opt.device)#device=self.cuda_devices[-1])

                outputs, hourglass_features = self.network(input2)
                '''
                outputs: torch.tensor, [12, 2, 16, 64, 64]
                hourglass_features: <dict>, (['out', 1, 'feature_1', 2, 'feature_2', 3, 'feature_3', 4, 'feature_4', 5, 'feature_5', 'penultimate'])
                
                '''
                outputs_go = outputs[:,1,:].detach().cpu().numpy() # type float32
                # print('outputs_go:', outputs_go.dtype)
                # exit()
                #Open
                #for k in range(12):
                #    self.vis_try_ht_pred(outputs_go[k], meta['index'][k])
                
                loss = heatmap_loss(outputs, target, self.n_stack)
                                
                '''
                o_c = outputs.cpu().detach().numpy()

                for k in range(12):
                    self.vis_try_ht_2(o_c[k], meta['index'][k])
                '''
                
                # Will clear the gradients of hourglass
                self.optimizer.zero_grad()
            
                
                learning_loss_ = loss.clone().detach()
                
                learning_loss_ = torch.mean(learning_loss_, dim=[1])
                #torch.Size([4]) tensor([4.5268e-04, 3.8438e-05, 4.6207e-05, 1.6311e-03], device='cuda:0')

                loss = (torch.mean(loss)) * self.train_hg_bool
                self.tb_writer.add_scalar('Train/Loss_batch', torch.mean(loss), global_step)

                loss.backward()
                # Train the learning loss network
                if self.train_learning_loss:
                    loss_learnloss = self.learning_loss(hourglass_features, learning_loss_, self.learnloss_margin, input2, e)
                    loss_learnloss.backward()
                    epoch_loss_learnloss.append(loss_learnloss.cpu().data.numpy())

                # Weight update
                self.optimizer.step()
                global_step += 1

                # Store the loss per batch
                epoch_loss.append(loss.cpu().data.numpy())

            epoch_loss = np.mean(epoch_loss)
            if self.train_learning_loss:
                epoch_loss_learnloss = np.mean(epoch_loss_learnloss)

            # Returns average validation loss per element
            if self.train_learning_loss:
                validation_loss_hg, validation_learning_loss = self.validation(e)
            else:
                validation_loss_hg = self.validation(e)
                validation_learning_loss = 0.0

            # Learning rate scheduler on the HourGlass validation loss
            self.scheduler.step(validation_loss_hg)

            # TensorBoard Summaries
            self.tb_writer.add_scalar('Train', torch.Tensor([epoch_loss]), global_step)
            self.tb_writer.add_scalar('Validation/HG_Loss', torch.Tensor([validation_loss_hg]), global_step)
            if self.train_learning_loss:
                self.tb_writer.add_scalar('Validation/Learning_Loss', torch.Tensor([validation_learning_loss]), global_step)

            # Save the model
            torch.save(self.network.state_dict(),
                       self.model_save_path.format("model_epoch_{}.pth".format(e + 1)))

            if self.train_learning_loss:
                torch.save(self.learnloss_network.state_dict(),
                           self.model_save_path.format("model_epoch_{}_learnloss.pth".format(e + 1)))

            # For resume training ONLY:
            # If learn_loss, then optimizer will have two param groups
            # Hence during load, ensure llal module is loaded/not loaded

            torch.save({'epoch': e + 1,
            'optimizer_load_state_dict': self.optimizer.state_dict(),
            'mean_loss': epoch_loss,
            
            'learn_loss': self.train_learning_loss},
            self.model_save_path.format("optim_epoch_{}.tar".format(e + 1)))
            torch.save({'epoch': e + 1,
                        'optimizer_load_state_dict': self.optimizer.state_dict(),
                        'mean_loss': epoch_loss,
                        'mean_loss_validation': {'HG': validation_loss_hg, 'LearningLoss': validation_learning_loss},
                        'learn_loss': self.train_learning_loss},
                        self.model_save_path.format("optim_epoch_{}.tar".format(e + 1)))

            # Save if best model
            if best_val_hg > validation_loss_hg:
                torch.save(self.network.state_dict(),
                           self.model_save_path.format("best_model.pth"))

                torch.save(self.learnloss_network.state_dict(),
                           self.model_save_path.format("best_model_learnloss_hg.pth"))

                best_val_hg = validation_loss_hg
                best_epoch_hg = e + 1

                torch.save({'epoch': e + 1,
                            'optimizer_load_state_dict': self.optimizer.state_dict(),
                            'mean_loss_train': epoch_loss,
                            'mean_loss_validation': {'HG': validation_loss_hg, 'LearningLoss': validation_learning_loss},
                            'learn_loss': self.train_learning_loss},
                           self.model_save_path.format("optim_best_model.tar"))

            if self.train_learning_loss:
                if best_val_learnloss > validation_learning_loss and validation_learning_loss != 0.0:
                    torch.save(self.learnloss_network.state_dict(),
                               self.model_save_path.format("best_model_learnloss_{}.pth".format(self.learnloss_obj)))

                    best_val_learnloss = validation_learning_loss
                    best_epoch_learnloss = e + 1


            print("Loss at epoch {}/{}: (train) {}\t"
                  "Learning Loss: (train) {}\t"
                  "(validation: HG) {}\t"
                  "(Validation: Learning Loss) {}\t"
                  "(Best Model) {}".format(
                e+1,
                self.epoch,
                epoch_loss,
                epoch_loss_learnloss,
                validation_loss_hg,
                validation_learning_loss,
                best_epoch_hg))

            loss_across_epochs.append(epoch_loss)
            validation_across_epochs.append(validation_loss_hg)

            # Save the loss values
            f = open(self.model_save_path.format("loss_data.txt"), "w")
            f_ = open(self.model_save_path.format("validation_data.txt"), "w")
            f.write("\n".join([str(lsx) for lsx in loss_across_epochs]))
            f_.write("\n".join([str(lsx) for lsx in validation_across_epochs]))
            f.close()
            f_.close()

        self.tb_writer.close()
        logging.info("Model training completed\nBest validation loss (HG): {}\tBest Epoch: {}"
                     "\nBest validation loss (LLAL): {}\tBest Epoch: {}".format(
            best_val_hg, best_epoch_hg, best_val_learnloss, best_epoch_learnloss))

    def validation(self, e):
        '''
        Validation loss
        :param e: (int) Epoch
        :return: (Float): Mean validation loss per batch for Hourglass and Learning Loss (if LL activated in inc_config file.)
        '''
        with torch.no_grad():
            # Stores the loss for all batches
            epoch_val_hg = []

            if self.train_learning_loss:
                epoch_val_learnloss = []

            self.network.eval()
            if self.train_learning_loss:
                self.learnloss_network.eval()

            # Augmentation only needed in Training
            #self.dataset_obj.input_dataset(validate=True)

            # Compute and store batch-wise validation loss in a list
            logging.info('Validation for epoch: {}'.format(e+1))
            for i, batch in enumerate(tqdm(self.val_dataloader)):
                input_val, target_val, meta_val = batch['input'], batch['target'], batch['meta'] 

                input_val = input_val.to(non_blocking=True, device=self.cuda_devices[-1])

                input_val = input_val.permute(0, 2, 3, 1) 

                outputs_val, hourglass_features_val = self.network(input_val)

                target_val = target_val.to(non_blocking=True, device=self.cuda_devices[-1])

                loss_val_hg = heatmap_loss(outputs_val, target_val, self.n_stack)

                learning_loss_val = loss_val_hg.clone().detach()
                learning_loss_val = torch.mean(learning_loss_val, dim=[1])

                loss_val_hg = torch.mean(loss_val_hg)
                epoch_val_hg.append(loss_val_hg.cpu().data.numpy())

                if self.train_learning_loss:
                    loss_val_learnloss = self.learning_loss(hourglass_features_val, learning_loss_val, self.learnloss_margin, input_val, e)
                    epoch_val_learnloss.append(loss_val_learnloss.cpu().data.numpy())

            print("Validation Loss HG at epoch {}/{}: {}".format(e+1, self.epoch, np.mean(epoch_val_hg)))

            if self.train_learning_loss:
                print("Validation Learning Loss at epoch {}/{}: {}".format(e+1, self.epoch, np.mean(epoch_val_learnloss)))
                return np.mean(epoch_val_hg), np.mean(epoch_val_learnloss)

            else:
                return np.mean(epoch_val_hg)

    def learning_loss(self, hg_encodings, true_loss, margin, gt_per_img, epoch):
        '''
        Learning Loss module
        Refer:
        1. "Learning Loss For Active Learning, CVPR 2019"
        2. "A Mathematical Analysis of Learning Loss for Active Learning in Regression, CVPRW 2021"
        :param hg_encodings: (Dict of tensors) Intermediate (Hourglass) and penultimate layer output of the Hourglass network
        :param true_loss: (Tensor of shape [Batch Size]) Loss computed from HG prediction and ground truth
        :param margin: (scalar) tolerance margin between predicted losses
        :param gt_per_img: (Tensor, shape [Batch Size]) Number of ground truth per image
        :param epoch: (scalar) Epoch, used in learning loss warm start-up
        :return: (Torch scalar tensor) Learning Loss
        '''

        # Concatenate the layers instead of a linear combination
        with torch.no_grad():
            if self.learnloss_original:
                # hg_depth == 4 means depth is {1, 2, 3, 4}. If we want depth 5, range --> (1, 4+2)
                # encodings = torch.cat([hg_encodings[depth] for depth in range(1, self.hg_depth + 2)], dim=-1)
                encodings = hg_encodings['penultimate']

            else:
                # No longer concatenating, will now combine features through convolutional layers
                encodings = torch.cat([hg_encodings['feature_5'].reshape(self.batch_size, hg_encodings['feature_5'].shape[1], -1),               # 64 x 64
                                       hg_encodings['feature_4'].reshape(self.batch_size, hg_encodings['feature_4'].shape[1], -1),               # 32 x 32
                                       hg_encodings['feature_3'].reshape(self.batch_size, hg_encodings['feature_3'].shape[1], -1),               # 16 x 16
                                       hg_encodings['feature_2'].reshape(self.batch_size, hg_encodings['feature_2'].shape[1], -1),               # 8 x 8
                                       hg_encodings['feature_1'].reshape(self.batch_size, hg_encodings['feature_1'].shape[1], -1)], dim=2)       # 4 x 4

        emperical_loss, encodings = self.learnloss_network(encodings)
        emperical_loss = emperical_loss.squeeze()
        
        #print('2D-Pred:')
        #print(emperical_loss.shape, emperical_loss)
        #print('2D-GT:')
        #print(true_loss.shape, true_loss)
        #ll_pred: torch.Size([4]) tensor([0.0235, 0.0234, 0.0233, 0.0235]
        #torch.Size([4]) tensor([1.9972e-04, 8.7234e-04, 5.5121e-05, 6.2852e-04]
        
        assert emperical_loss.shape == true_loss.shape, "Mismatch in Batch size for true and emperical loss"

        with torch.no_grad():
            # Scale the images as per the number of joints
            # To prevent DivideByZero. PyTorch does not throw an exception to DivideByZero
            gt_per_img = torch.sum(gt_per_img, dim=1)

            #'gt_per', gt_per_img.shape, gt_per_img) [B, 256, 3]
            
            gt_per_img += 0.1
            if self.learnloss_obj == 'prob':
                #print('Error', true_loss.shape) #[4]
                #print('Error2: ', gt_per_img.shape) #[4, 256, 3]
                true_loss = true_loss / gt_per_img
            # Splitting into pairs: (i, i+half)
            half_split = true_loss.shape[0] // 2

            true_loss_i = true_loss[: half_split]
            true_loss_j = true_loss[half_split: 2 * half_split]

        emp_loss_i = emperical_loss[: (emperical_loss.shape[0] // 2)]
        emp_loss_j = emperical_loss[(emperical_loss.shape[0] // 2): 2 * (emperical_loss.shape[0] // 2)]

        # Loss according to CVPR '19
        if self.learnloss_obj == 'pair':
            loss_sign = torch.sign(true_loss_i - true_loss_j)
            loss_emp = (emp_loss_i - emp_loss_j)

            # Learning Loss objective
            llal_loss = torch.max(torch.zeros(half_split, device=loss_sign.device), (-1 * (loss_sign * loss_emp)) + margin)

        # Loss according to CVPR '21
        elif self.learnloss_obj == 'prob':
            with torch.no_grad():
                true_loss_ = torch.cat([true_loss_i.reshape(-1, 1), true_loss_j.reshape(-1, 1)], dim=1)
                true_loss_scaled = true_loss_ / torch.sum(true_loss_, dim=1, keepdim=True)

            emp_loss_ = torch.cat([emp_loss_i.reshape(-1, 1), emp_loss_j.reshape(-1, 1)], dim=1)
            emp_loss_logsftmx = torch.nn.LogSoftmax(dim=1)(emp_loss_)
            llal_loss = torch.nn.KLDivLoss(reduction='batchmean')(input=emp_loss_logsftmx, target=true_loss_scaled)

        else:
            raise NotImplementedError('Currently only "pair" or "prob" supported. ')

        if self.learnloss_warmup <= epoch:
            return torch.mean(llal_loss)
        else:
            return 0.0 * torch.mean(llal_loss)

    def vis_merge(self, info, meta_info, target, file_name):
        '''
        info: (torch.Tensor)
        '''
        mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
        print('image index: ', meta_info['index'][0]+1)
        input2 = info[0].permute(1, 2, 0) 
        input2 = (input2*std + mean) # image 的matric

        input2_array = (input2.numpy()*255).astype(np.uint8)
        #print(input2_array.dtype)
        #cv2.imshow('image',input2_array)
        cv2.imwrite('../debug_vis/joints/image_img.png', input2_array) # RGB

        image_to_write = cv2.cvtColor(input2_array, cv2.COLOR_RGB2BGR) #cv2.COLOR_BGR2RGB
        cv2.imwrite('../debug_vis/joints/image_cvt.png', image_to_write) # RGB
        #cv2.imwrite('../debug_vis/joints/image.png', image_to_write)
        #cv2.waitKey(0)
    
        bottom_pic = input2_array
        #bottom_pic = cv2.resize(image_to_write, (64, 64), interpolation = cv2.INTER_AREA)

        b =torch.zeros((64, 64))
        for i in range(16): 
            b=target[0][i]
        
            b_array = (b.numpy()*255).astype(np.uint8)
            #print(b_array.dtype)
            heatmap = cv2.applyColorMap(b_array, cv2.COLORMAP_HOT)
            top_pic = cv2.resize(heatmap, (256, 256), interpolation = cv2.INTER_AREA)
            print(top_pic.shape)
            print(bottom_pic.shape)
            overlapping_pic = cv2.addWeighted(bottom_pic, 0.6, top_pic, 0.4, 0) #(256, 256, 3) and unit8
            cv2.imwrite('../debug_vis/joints/image_merge_joints_{}.png'.format(i), overlapping_pic)
            
            #plt.savefig("../debug_vis/joints/image_{}_joint_{}.png".format(file_name, i))

        return print("Merge!")

    def vis_try_ht(self, info, index):
        # heatmap # info放target
        b =torch.zeros((64, 64)).to('cuda:0')
        for i in range(15): 
            b+=info[i]
        b=b.to('cpu')
        plt.imshow(b, cmap='magma')
        plt.savefig("../1115_debug_vis/Training_GT_heatmap_{}.png".format(index))
        return print("Done with vis heatmap(GT) on Training")

    def vis_try_ht_pred(self, info, index):
        # heatmap # info放target
        b =torch.zeros((64, 64))
        for i in range(15): 
            b=info[i]
            
            plt.imshow(b, cmap='magma')
            plt.savefig("../1115_debug_vis_pred/Training_pred_heatmap_{}_{}.png".format(index, i))
        return print("Done with vis heatmap(Pred) on Training")

    def vis_try_ht_2(self, info, index):
        # heatmap # info放target
        b =torch.zeros((64, 64))
        for i in range(15): 
            b+=info[0][i]
        plt.imshow(b, cmap='magma')
        plt.savefig("../debug_vis/Training_Pred_heatmap_{}.png".format(index))
        return print("Done with vis heatmap on Training")

    def vis_try(self, info, meta_info):
        #畫input(crop image)
        mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
        print('image index: ', meta_info['index'][0]+1)
        input2 = info[0].permute(1, 2, 0) 
        input2 = (input2*std + mean)
        plt.imshow(input2)
        plt.savefig("../debug_vis/Training_input_crop.png")
        
        return print("Done with vis input on Training")


# Inferenc the model
class Test(object):
    def __init__(self, network, dataset_obj, opt):
            
        self.network = network
        self.dataset_obj = dataset_obj
        self.opt = opt
        
        self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    def is_image(self,file_name):
        image_ext = ['jpg', 'jpeg', 'png']
        ext = file_name[file_name.rfind('.') + 1:].lower()
        return ext in image_ext

    def get_preds(self, hm, return_conf=False):
        assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
        h = hm.shape[2]
        w = hm.shape[3]
        hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
        idx = np.argmax(hm, axis = 2)
        
        preds = np.zeros((hm.shape[0], hm.shape[1], 2))
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                preds[i, j, 0], preds[i, j, 1] = idx[i, j] % w, idx[i, j] / w
        if return_conf:
            conf = np.amax(hm, axis = 2).reshape(hm.shape[0], hm.shape[1], 1)
            return preds, conf
        else:
            return preds
            
    def demo_image(self, image, model, opt):
        s = max(image.shape[0], image.shape[1]) * 1.0
        c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
        trans_input = get_affine_transform(
            c, s, 0, [opt.input_w, opt.input_h])
        inp = cv2.warpAffine(image, trans_input, (opt.input_w, opt.input_h),
                                flags=cv2.INTER_LINEAR)
        inp = (inp / 255. - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        inp = torch.from_numpy(inp).to(opt.device)
        out = model(inp)[-1]
        pred = self.get_preds(out['hm'].detach().cpu().numpy())[0]
        pred = transform_preds(pred, c, s, (opt.output_w, opt.output_h))
        #pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(), 
                               # out['depth'].detach().cpu().numpy())[0]
        
        debugger = Debugger()
        debugger.add_img(image)
        debugger.add_point_2d(pred, (255, 0, 0))
        #debugger.add_point_3d(pred_3d, 'b')
        debugger.show_all_imgs(pause=False)
        #debugger.show_3d()

    def eval(self):
        opt.heads['depth'] = opt.num_output
        if opt.load_model == '':
            opt.load_model = '../Experiments/1012_Test_713/model_checkpoints/best_model.pth'
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
        else:
            opt.device = torch.device('cpu')
        print('Load model')
        model = self.network.to(opt.device)
        model.eval()
        
        if os.path.isdir(opt.demo):
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                if self.is_image(file_name):
                    image_name = os.path.join(opt.demo, file_name)
                    print('Running {} ...'.format(image_name))
                    image = cv2.imread(image_name)
                    self.demo_image(image, model, opt)
                elif self.is_image(opt.demo):
                    print('Running {} ...'.format(opt.demo))
                    image = cv2.imread(opt.demo)
                    self.demo_image(image, model, opt)




def vis_try(info, meta_info):
    #畫input(crop image)
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    print('image index: ', meta_info['index'][0]+1)
    input2 = info[0].permute(1, 2, 0) 
    input2 = (input2*std + mean)
    plt.imshow(input2)
    plt.savefig("../debug_vis/dataloader_input_crop.png")
    
    return print("Done with vis input")

    # 畫出crop 的東西
    # import matplotlib.pyplot as plt
    # b =torch.zeros((64, 64))
    # for i in range(15): 
    #     b+=meta['pts_crop'][0][i]
    
    # plt.imshow(b, cmap='magma')
    # plt.savefig("mygraph_crop.png")
    # exit()

def vis_try_ht(info):
    # heatmap # info放target
    import matplotlib.pyplot as plt
    b =torch.zeros((64, 64))#.to('cuda:0')
    for i in range(15): 
        b+=info[0][i]
    # b=b.to('cpu')
    plt.imshow(b, cmap='magma')
    plt.savefig("../debug_vis/dataloader_target_heatmap2.png")
    return print("Done with vis heatmap(GT)")

def vis_try_ht_pred(info):
    # heatmap # info放target
    import matplotlib.pyplot as plt
    b =torch.zeros((64, 64)).to('cuda:0')
    for i in range(15): 
        b+=info[0][i]
    b=b.to('cpu')
    plt.imshow(b, cmap='magma')
    plt.savefig("../debug_vis_1/prediction.png")
    return print("Done with vis prediction")



def vis_try_2(info, meta_info):
    #畫input(crop image)
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    print('image index: ', meta_info['index'][0]+1)
    input2 = info[0].permute(1, 2, 0) 
    input2 = (input2*std + mean)
    plt.imshow(input2)
    plt.savefig("../debug_vis/joints/dataloader_input_crop.png")
    
    return print("Done with vis input")

def vis_joint_name(info, file_name):
    # heatmap # info放target
    b =torch.zeros((64, 64))
    for i in range(15): 
        b=info[0][i]
        plt.imshow(b,cmap='magma')
        plt.savefig("../debug_vis/joints/image_{}_joint_{}.png".format(file_name, i))
    return print("Joint_names")

def vis_merge(info, meta_info, target, file_name):
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    print('image index: ', meta_info['index'][0]+1)
    input2 = info[0].permute(1, 2, 0) 
    input2 = (input2*std + mean) # image 的matric

    input2_array = (input2.numpy()*255).astype(np.uint8)
    #print(input2_array.dtype)
    #cv2.imshow('image',input2_array)
    cv2.imwrite('../debug_vis/joints/image_img.png', input2_array) # RGB

    image_to_write = cv2.cvtColor(input2_array, cv2.COLOR_RGB2BGR) #cv2.COLOR_BGR2RGB
    cv2.imwrite('../debug_vis/joints/image_cvt.png', image_to_write) # RGB
    #cv2.imwrite('../debug_vis/joints/image.png', image_to_write)
    #cv2.waitKey(0)
   
    bottom_pic = input2_array
    #bottom_pic = cv2.resize(image_to_write, (64, 64), interpolation = cv2.INTER_AREA)

    b =torch.zeros((64, 64))
    for i in range(16): 
        b=target[0][i]
       
        b_array = (b.numpy()*255).astype(np.uint8)
        #print(b_array.dtype)
        heatmap = cv2.applyColorMap(b_array, cv2.COLORMAP_HOT)
        top_pic = cv2.resize(heatmap, (256, 256), interpolation = cv2.INTER_AREA)
        print(top_pic.shape)
        print(bottom_pic.shape)
        overlapping_pic = cv2.addWeighted(bottom_pic, 0.6, top_pic, 0.4, 0) #(256, 256, 3) and unit8
        cv2.imwrite('../debug_vis/joints/image_merge_joints_{}.png'.format(i), overlapping_pic)
        
        #plt.savefig("../debug_vis/joints/image_{}_joint_{}.png".format(file_name, i))

    return print("Merge!")

def vis_try_ht_pred_2(info, file_name):
    # heatmap # info放target
    b =torch.zeros((64, 64))
    for i in range(15): 
        b=info[0][i]
        #print(b.shape) # 64, 64
        #print(image.shape)
        #imggg = cv2.addWeighted(image, 0.4, b, 0.6, 0)
        #cv2.imwrite("../debug_vis/prediction_imagesss_{}.png".format(i), imggg)
        plt.imshow(b,cmap='magma')
        plt.savefig("../debug_vis/prediction_{}_images_ht_{}.png".format(file_name, i))
    return print("Done with vis prediction")

class Test2(object):

    '''
    1) inference_dict
    2) compute PCKh 
    
    '''
    def __init__(self, network, dataset_obj, conf):

        self.dataset_obj = dataset_obj
        self.network = network
        self.conf = conf
        self.model_save_path = conf.model_save_path

        self.ind_to_jnt = {0: 'rfoot', 1: 'rknee', 2: 'rhip', 3: 'lhip', 4: 'lknee', 5: 'lfoot', 6: 'torso', 7: 'chest',
                    8: 'Neck', 9: 'Head', 10: 'rhand', 11: 'relb', 12: 'rsho', 13:'lsho', 14: 'lelb', 15:'lhand'}

        # self.ind_to_jnt = {0: 'head', 1: 'neck', 2: 'lsho', 3: 'lelb', 4: 'lwri', 5: 'rsho', 6: 'relb', 7: 'rwri',
        #                    8: 'lhip', 9: 'lknee', 10: 'lankl', 11: 'rhip', 12: 'rknee', 13: 'rankl'}
        # if conf.num_hm == 16:
        #             self.ind_to_jnt[14] = 'pelvis'
        #             self.ind_to_jnt[15] = 'thorax'

        #self.torch_dataloader = torch.utils.data.DataLoader(self.dataset_obj, conf.batch_size, shuffle=False,
                                                            #num_workers=0)

    def get_preds(self, hm, return_conf=False):
        assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
        h = hm.shape[2]
        w = hm.shape[3]
        hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
        idx = np.argmax(hm, axis = 2)
        
        preds = np.zeros((hm.shape[0], hm.shape[1], 2))
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                preds[i, j, 0], preds[i, j, 1] = idx[i, j] % w, idx[i, j] / w
        if return_conf:
            conf = np.amax(hm, axis = 2).reshape(hm.shape[0], hm.shape[1], 1)
            return preds, conf
        else:
            return preds # shape: [batch, joint, 0:1]

    def inference(self):
        self.network.eval()
        print('Start inference')

        image_index = torch.tensor([]) 
        image_nor = torch.tensor([]) 
        output_xy = np.empty((1, 16, 2))
        target_xy = np.empty((1, 16, 2))
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.dataset_obj)):
                
                '''
                input: torch.Size([32, 3, 256, 256])
                target: torch.Size([32, 16, 64, 64])
                meta_val: torch.size[1]

                outputs_val2:  <class 'torch.Tensor'> torch.Size([1, 16, 64, 64])
                target_val:  <class 'torch.Tensor'> torch.Size([1, 16, 64, 64])

                pred_val: (np.array) (1, 16, 2)
                target_val2: (np.narry) (1, 16, 2)
                meta_val['index][0]+1: image的名字

                '''
                input_val, target_val, meta_val = batch['input'], batch['target'], batch['meta'] 
                input_val = input_val.to(non_blocking=True, device=opt.device)
                input_val = input_val.permute(0, 2, 3, 1) 
                outputs_val, hourglass_features_val = self.network(input_val) # 算出來的heatmap

                outputs_val2 = outputs_val.cpu().detach().numpy().copy()
                outputs_val2 = outputs_val.mean(axis= 1)

                pred_val = self.get_preds(outputs_val2.detach().cpu().numpy()) #(1, 16, 2)

                target_val = target_val.to(non_blocking=True, device=opt.device)
                target_val2 = self.get_preds(target_val.detach().cpu().numpy())                

                #先關掉
                # self.vis_2D_pred(pred_val[0], meta_val['index'])
                # self.vis_2D_gt(target_val2[0], meta_val['index'])


                # 沒有進到get_preds之前的heatmap出來的樣子
                # self.vis_try_ht_pred(outputs_val2[0], meta_val['index'])
                # self.vis_try_ht(target_val[0], meta_val['index'])
                
                # cv2.imshow('peek',input_val[1].cpu().numpy())
                # cv2.waitKey(0)
                # exit()
             
                #image_index = torch.cat((image_index, int(meta_val['index'][0].cpu().numpy())+1))

                
                #print('pred_val', pred_val.shape, type(pred_val)) # (1, 16 ,2)target_xy
                #print('target_xy', target_xy.shape, type(target_xy))
                
                image_index = torch.cat((image_index, meta_val['index']+1), dim=0)
                output_xy= np.append(output_xy, pred_val, axis=0)
                target_xy= np.append(target_xy, target_val2, axis=0)# 第一個不要看...
                image_nor = torch.cat((image_nor, meta_val['normalizer']), dim=0)
                
                # output_xy= np.append([output_xy], pred_val, axis=0)
                # target_xy= np.append([target_xy], target_val2, axis=0)
                #target_xy .concatenate(target_val2)
                # output_xy = torch.cat((output_xy, pred_val.copy()), dim=0)
                # target_xy = torch.cat((target_xy, target_val2.copy()), dim=0)
 
        #model_inference={'image':image_index.copy(), 'output':output_xy.copy(), 'target':target_xy.copy()}
        model_inference={'img_name':image_index.clone(),'output':output_xy.copy(), 'target':target_xy.copy(), 'img_nor':image_nor.clone()}
        
        # import pickle
        # with open('1029.pickle', 'wb') as handle:
        #     pickle.dump(model_inference, handle, protocol=pickle.HIGHEST_PROTOCOL) #https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
        
        return model_inference

    def vis_2D_gt(self, info, index):
        for i in range(15):
            plt.scatter(info[i][0], info[i][1])
        plt.savefig("../1117_2D_vis_gt/gt_{}.png".format(index))
        return print("Done with gt")

    def vis_2D_pred(self, info, index):
        for i in range(15):
            plt.scatter(info[i][0], info[i][1])
        plt.savefig("../1117_2D_vis_pred/pred_{}.png".format(index))
        return print("Done with pred")
        
    def vis_try_ht(self, info, index):
        # heatmap # info放target
        b = torch.zeros((64, 64)).to('cuda:0')
        for i in range(15): 
            b+=info[i]
        b=b.to('cpu')
        plt.imshow(b)
        plt.savefig("../1117_debug_vis/Training_GT_heatmap_{}.png".format(index))
        return print("Done with vis heatmap(GT) on Training")

    def vis_try_ht_pred(self, info, index):
        # heatmap # info放target
        b =torch.zeros((64, 64)).to('cuda:0')
        for i in range(15): 
            b+=info[i]
        b=b.to('cpu')    
        plt.imshow(b)
        plt.savefig("../1117_debug_vis_pred/Training_pred_heatmap_{}.png".format(index))
        return print("Done with vis heatmap(Pred) on Training")

    def keypoint(self, infer):
        '''
        Scale the heatmap joints from acual U, V on unscaled image

        infer: (dict)['image_name', 'output', 'target'] 
        '''
        #image = infer['image'] 
        image_name = infer['img_name']
        gt = infer['target']
        output = infer['output']
        nor = infer['img_nor']

        csv_columns = ['name', 'joint', 'uv', 'normalizer']
        gt_csv = []
        pred_csv = []
        
        # print('image_name: ', len(image_name), image_name.shape, type(image_name))
        # print('gt: ', len(gt), gt.shape, type(gt))
        # print('output: ', len(output), output.shape, type(output))

        for i in range(len(gt)-1):
            for jnt in range(15):
                gt_entry = {
                    'name': '{}.jpg'.format(int(image_name[i])),
                    'joint': self.ind_to_jnt[jnt], 
                    'uv': gt[i+1][jnt],
                    'normalizer':np.float32(nor[i])
                }
                pred_entry = {
                    'name': '{}.jpg'.format(int(image_name[i])),
                    'joint': self.ind_to_jnt[jnt], 
                    'uv': output[i+1][jnt],
                    'normalizer':np.float32(nor[i])
                }
                gt_csv.append(gt_entry)
                pred_csv.append(pred_entry)

        pred_csv = pd.DataFrame(pred_csv, columns=csv_columns)
        gt_csv = pd.DataFrame(gt_csv, columns=csv_columns)

        pred_csv.sort_values(by='name', ascending=True, inplace=True)
        gt_csv.sort_values(by='name', ascending=True, inplace=True)

        pred_csv.to_csv(self.model_save_path.format("pred.csv"), index=False)
        gt_csv.to_csv(self.model_save_path.format("gt.csv"), index=False)
        
        logging.info('Pandas dataframe saved successfully.')
        
        return gt_csv, pred_csv
        

    def distance(self, gt, pred, normalizer):
        
        dist_ = np.linalg.norm(gt[:2].astype(np.float32)- pred[:2].astype(np.float32))
        dist_/= normalizer
        return dist_
    
    def percent_correct_keypoint(self, pred_df, gt_df, conf, joints):
        joint_names = joints

        distance_df = pd.DataFrame()
        distance_df['name'] = gt_df.name
        distance_df['joint'] = gt_df.joint
        distance_df['gt_uv'] = gt_df.uv
        distance_df['pred_uv'] = pred_df.uv
        distance_df['normalizer'] = gt_df.normalizer
        
        distance_df['distance'] = distance_df.apply(lambda row: self.distance(row.gt_uv, row.pred_uv, row.normalizer), axis=1)

        threshold = np.linspace(0, 1, num=20)

        pck_dict = {}
        pck_dict['threshold'] = threshold

        pck_dict['average'] = np.zeros_like(threshold)

        num_joints = len(joint_names)

        for jnt in tqdm(joint_names):
            pck_dict[jnt] = []

            distance_sub = distance_df[distance_df.joint==jnt]
            distance_sub = distance_sub[distance_df.distance >= 0.0]
            
            total_gt = len(distance_sub.index)

            for th in threshold:
                distance_th = distance_sub[distance_sub.distance< th]
                pck_dict[jnt].append(len(distance_th.index)/(total_gt+ 1e-5))

            pck_dict['average'] += np.array(pck_dict[jnt])
        pck_dict['average']/=num_joints
        pck_csv = pd.DataFrame(pck_dict)
        return pck_csv


    def compute_metric(self, gt_df, pred_df):
        print('Generate evaluation metrics----')
        pred_ = pred_df
        gt_ = gt_df
        pck_df = self.percent_correct_keypoint(pred_df=pred_, gt_df=gt_, conf=self.conf,
                                               joints=list(self.ind_to_jnt.values()))

        pck_df.to_csv(self.model_save_path.format("PCKh.csv"), index=False)
        print("Finally")

    
    def eval(self):
        '''
        Control flow to obtain predictions and corresponding metrics from Test()
        '''
        model_inference = self.inference()
        gt_csv, pred_csv = self.keypoint(model_inference)
        self.compute_metric(gt_df=gt_csv, pred_df=pred_csv)

class Test3(object):
    '''
    因為training_dataset 跟 validation dataset不同
    '''
    def __init__(self, network, dataset_obj, conf):
        self.dataset_obj = dataset_obj
        self.conf = conf
        self.network = network
        self.ind_to_jnt = {0: 'rfoot', 1: 'rknee', 2: 'rhip', 3: 'lhip', 4: 'lknee', 5: 'lfoot', 6: 'torso', 7: 'chest',
            8: 'Neck', 9: 'Head', 10: 'rhand', 11: 'relb', 12: 'rsho', 13:'lsho', 14: 'lelb', 15:'lhand'}
        self.model_save_path = conf.model_save_path

    def get_preds(self, hm, return_conf=False):
        assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
        h = hm.shape[2]
        w = hm.shape[3]
        hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
        idx = np.argmax(hm, axis = 2)
        
        preds = np.zeros((hm.shape[0], hm.shape[1], 2))
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                preds[i, j, 0], preds[i, j, 1] = idx[i, j] % w, idx[i, j] / w
        if return_conf:
            conf = np.amax(hm, axis = 2).reshape(hm.shape[0], hm.shape[1], 1)
            return preds, conf
        else:
            return preds # shape: [batch, joint, 0:1]

    def inference(self):
        self.network.eval()
        print('Start inference')

        image_index = torch.tensor([]) 
        image_nor = torch.tensor([]) 
        output_xy = np.empty((1, 16, 2))
        target_xy = np.empty((1, 16, 2))
        
        with torch.no_grad():
            for (inp, out, meta, image)in tqdm(self.dataset_obj):
                input2 = inp.permute(0, 2, 3, 1)
                input2 = input2.cuda(device=opt.device, non_blocking=True)
                target = out.cuda(device=opt.device, non_blocking=True)
                outputs, hourglass_features = self.network(input2)
                '''
                input2:  <class 'torch.Tensor'> torch.Size([1, 256, 256, 3])
                target:  <class 'torch.Tensor'> torch.Size([1, 16, 64, 64])
                name:  <class 'torch.Tensor'> torch.Size([1])
                output:  <class 'torch.Tensor'> torch.Size([1, 2, 16, 64, 64])
                meta: 'index', 'center', 'scale', 'gt_3d', 'pts_crop', 'normalizer'])
                '''
                pred_val = self.get_preds(outputs[:,0, :].detach().cpu().numpy()) #(1, 16, 2)
                target_val2 = self.get_preds(target.detach().cpu().numpy())   


                image_index = torch.cat((image_index, meta['index']+1), dim=0)
                output_xy= np.append(output_xy, pred_val, axis=0)
                target_xy= np.append(target_xy, target_val2, axis=0)# 第一個不要看...
                image_nor = torch.cat((image_nor, meta['normalizer'] ), dim=0)

        model_inference={'img_name':image_index.clone(),'output':output_xy.copy(), 'target':target_xy.copy(), 'img_nor':image_nor.clone()}
        return model_inference

    def keypoint(self, infer):
        '''
        Scale the heatmap joints from acual U, V on unscaled image

        infer: (dict)['image_name', 'output', 'target'] 
        '''
        #image = infer['image'] 
        image_name = infer['img_name']
        gt = infer['target']
        output = infer['output']
        nor = infer['img_nor']

        csv_columns = ['name', 'joint', 'uv', 'normalizer']
        #csv_columns = ['name', 'joint', 'uv']
        gt_csv = []
        pred_csv = []
        
        # print('image_name: ', len(image_name), image_name.shape, type(image_name))
        # print('gt: ', len(gt), gt.shape, type(gt))
        # print('output: ', len(output), output.shape, type(output))

        for i in range(len(gt)-1):
            for jnt in range(15):
                gt_entry = {
                    'name': '{}.jpg'.format(int(image_name[i])),
                    'joint': self.ind_to_jnt[jnt], 
                    'uv': gt[i+1][jnt],
                    'normalizer':np.float32(nor[i])
                }
                pred_entry = {
                    'name': '{}.jpg'.format(int(image_name[i])),
                    'joint': self.ind_to_jnt[jnt], 
                    'uv': output[i+1][jnt],
                    'normalizer':np.float32(nor[i])
                }
                gt_csv.append(gt_entry)
                pred_csv.append(pred_entry)

        pred_csv = pd.DataFrame(pred_csv, columns=csv_columns)
        gt_csv = pd.DataFrame(gt_csv, columns=csv_columns)

        pred_csv.sort_values(by='name', ascending=True, inplace=True)
        gt_csv.sort_values(by='name', ascending=True, inplace=True)

        pred_csv.to_csv(self.model_save_path.format("pred.csv"), index=False)
        gt_csv.to_csv(self.model_save_path.format("gt.csv"), index=False)
        
        logging.info('Pandas dataframe saved successfully.')
        
        return gt_csv, pred_csv
        

    def distance(self, gt, pred, normalizer):
        
        dist_ = np.linalg.norm(gt[:2].astype(np.float32)- pred[:2].astype(np.float32))
        dist_/= normalizer
        return dist_
    
    def percent_correct_keypoint(self, pred_df, gt_df, conf, joints):
        joint_names = joints

        distance_df = pd.DataFrame()
        distance_df['name'] = gt_df.name
        distance_df['joint'] = gt_df.joint
        distance_df['gt_uv'] = gt_df.uv
        distance_df['pred_uv'] = pred_df.uv
        distance_df['normalizer'] = gt_df.normalizer
        
        distance_df['distance'] = distance_df.apply(lambda row: self.distance(row.gt_uv, row.pred_uv, row.normalizer), axis=1)

        threshold = np.linspace(0, 1, num=20)

        pck_dict = {}
        pck_dict['threshold'] = threshold

        pck_dict['average'] = np.zeros_like(threshold)

        num_joints = len(joint_names)

        for jnt in tqdm(joint_names):
            pck_dict[jnt] = []

            distance_sub = distance_df[distance_df.joint==jnt]
            distance_sub = distance_sub[distance_df.distance >= 0.0]
            
            total_gt = len(distance_sub.index)

            for th in threshold:
                distance_th = distance_sub[distance_sub.distance< th]
                pck_dict[jnt].append(len(distance_th.index)/(total_gt+ 1e-5))

            pck_dict['average'] += np.array(pck_dict[jnt])
        pck_dict['average']/=num_joints
        pck_csv = pd.DataFrame(pck_dict)
        return pck_csv


    def compute_metric(self, gt_df, pred_df):
        print('Generate evaluation metrics----')
        pred_ = pred_df
        gt_ = gt_df
        pck_df = self.percent_correct_keypoint(pred_df=pred_, gt_df=gt_, conf=self.conf,
                                               joints=list(self.ind_to_jnt.values()))

        pck_df.to_csv(self.model_save_path.format("PCKh.csv"), index=False)
        print("Finally")

    
    def eval(self):
        '''
        Control flow to obtain predictions and corresponding metrics from Test()
        '''
        model_inference = self.inference()
        gt_csv, pred_csv = self.keypoint(model_inference)
        self.compute_metric(gt_df=gt_csv, pred_df=pred_csv)

class Train_3D(object):
    def __init__(self, network_2D, network_3D, learnloss_network, hyperparameters, dataset_obj, val_dataset_obj, tb_writer, conf, opt):
        '''
        network_2D: best pth from 2D training
        network_3D: 2D to 3D 
        dataset_obj: h36m
        '''
        self.network_2D = network_2D
        self.network_3D = network_3D

        self.learnloss_network = learnloss_network
        self.learnloss_margin = conf.learning_loss_margin
        self.learnloss_warmup = conf.learning_loss_warmup
        self.learnloss_original = conf.learning_loss_original
        self.learnloss_obj = conf.learning_loss_obj

        self.hyperparameters = hyperparameters
        self.optimizer = hyperparameters['optimizer']
        self.start_epoch = hyperparameters['start_epoch']
        self.torch_dataloader = dataset_obj
        self.val_dataset_obj = val_dataset_obj
        self.tb_writer = tb_writer                                           # Tensorboard writer
        self.conf = conf
        self.opt = opt

        self.tb_writer = tb_writer
        self.train_learning_loss = conf.train_learning_loss
        self.model_save_path = conf.model_save_path
        # TODO: save the tbwriter 

        self.total_steps = conf.total_steps
        self.epochs_3D = conf.epochs_3D
        self.batch_size_3D = conf.batch_size_3D

        """
        index_2D = {0: 'rfoot', 1: 'rknee', 2: 'rhip', 3: 'lhip', 4: 'lknee', 5: 'lfoot', 6: 'torso', 7: 'chest',
            8: 'Neck', 9: 'Head', 10: 'rhand', 11: 'relb', 12: 'rsho', 13:'lsho', 14: 'lelb', 15:'lhand', 16:'b_torso'}

        index_3D = {0:'b_torso', 1:'lhip', 2:'lknee', 3:'lfoot', 4:'rhip', 5:'rknee', 6:'rfoot', 7:'c_torso', 
            8:'u_torso', 9:'neck', 10:'head', 11:'rsho', 12:'relb', 13:'rhand', 14:'lsho', 15:'lelb', 16:'lhand'}
        
        self.idx_2D_to_3D reason
        """
        self.idx_2D_to_3D = [16, 3, 4, 5, 2, 1, 0, 7, 6, 8, 9, 12, 11, 10, 13, 14, 15]
    
        # ------------ Prepare Variable------------
        # img_path = model_path + "img/" #YY: in main calls the data path
        # save_path = model_path + "save/"

    def get_preds(self, hm, return_conf=False):
        assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
        h = hm.shape[2]
        w = hm.shape[3]
        hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
        idx = np.argmax(hm, axis = 2)
        
        preds = np.zeros((hm.shape[0], hm.shape[1], 2))
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                preds[i, j, 0], preds[i, j, 1] = idx[i, j] % w, idx[i, j] / w
        if return_conf:
            conf = np.amax(hm, axis = 2).reshape(hm.shape[0], hm.shape[1], 1)
            return preds, conf
        else:
            return preds # shape: [batch, joint, 0:1]

    def vis_3D(self, info):
        info = info.cpu().numpy()
        fig = plt.figure()
        #ax = fig.gca(projection='3d')

        for i in range(info.shape[0]):
            plt.scatter(info[i][0], info[i][1])
            plt.savefig("gt_3D_{}.png".format(i))
        return print("plot the 3D gt")

    def train_model_3D(self):
        print('Start training 3D model')
        print('Total steps - {}\t Batch Size - {}'.format(self.total_steps, self.batch_size_3D))
        
        best_val_network_3D = np.inf
        best_val_learnloss_3D = np.inf 
        best_epoch_network_3D = -1 
        best_epoch_learnloss_3D = -1 
        global_step =0

        # # Initial parameters
        best_val_hg = np.inf
        best_val_learnloss = np.inf
        best_epoch_hg = -1 
        best_epoch_learnloss = -1 
        global_step = 0

        loss_across_epochs = []
        validation_across_epochs = []

        
        total_loss_list, loss_func_list, kl_loss_list = [], [], [] # Calvin
        train_record = {"total_loss":[], "loss_func":[], "kl_loss": []} # Calvin

        for e in range(self.start_epoch, self.epochs_3D):

            epoch_loss = []
            epoch_loss_learnloss = []

            #self.network_2D.train()
            self.network_3D.train()

            if self.train_learning_loss:
                self.learnloss_network.train()

            print('Training for epoch: {}'.format(e+1))

            # total_loss_list, loss_func_list, kl_loss_list = [], [], [] # Calvin
            # train_record = {"total_loss":[], "loss_func":[], "kl_loss": []} # Calvin

            for (inp, out, meta, images) in tqdm(self.torch_dataloader):
                '''
                meta = {'index' : self.idxs[index], 'center' : c, 'scale' : s, 
                        'gt_3d': gt_3d, 'pts_crop': pts_crop, 'normalizer':normalizer}

                {'input': inp, 'target': out, 'meta': meta, 
                        'reg_target': reg_target, 'reg_ind': reg_ind, 'reg_mask': reg_mask, 'image':inp_copy}#, 'ht':out}
                '''
                
                input_2D = inp.permute(0, 2, 3, 1)
                input_2D = input_2D.cuda(device=self.opt.device, non_blocking=True)
                target_3D = meta['gt_3d'].cuda(device=self.opt.device, non_blocking=True) #x #meta['gt_3d']
                pred_2D, hourglass_features = self.network_2D(input_2D) #c

                #D
                # print('Index')
                # print(meta['index'][0])
                # print('training 2D-GT')
                # out = self.get_preds(out)
                # print(out[0])

                # print('training 2D-Pred')
                # pred_2D = pred_2D.cpu().detach().numpy().mean(axis= 1)
                # pred_2D_max = self.get_preds(pred_2D) #(1, 16, 2)
                
                # print(torch.tensor(pred_2D_max[0]))
                # #check 3D index 
                # print('training-3D-GT')
                # print(target_3D[0])
                # exit()
                #D
                '''
                input_2D:  <class 'torch.Tensor'> torch.Size([B, 256, 256, 3])
                target_3D:  <class 'torch.Tensor'> torch.Size([B, 17, 3])
                pred_2D: <class 'torch.Tensor'> torch.Size([B, 2, 16, 64, 64])
                pred_2D_max: torch.Size([B, 16, 2]) torch.float32
                target_3D: torch.Size([12, 16, 3]) torch.float32
                pred_3D: torch.Size([12, 48])
                '''
                pred_2D = pred_2D.cpu().detach().numpy().mean(axis= 1)
                pred_2D_max = self.get_preds(pred_2D) #(B, 16, 2)
                pred_2D_max = pred_2D_max.astype(np.float32)
                

                pred_2D_train = []
                # 加入element and index
                for i in range(pred_2D_max.shape[0]):
                    b_torse = (pred_2D_max[i][2]+ pred_2D_max[i][3])/2
                    pred_2D_train.append(np.concatenate((pred_2D_max[i], [b_torse]), axis=0)[self.idx_2D_to_3D])
                pred_2D_train = np.array(pred_2D_train)
                # 加入element and index

                pred_2D_train = torch.tensor(pred_2D_train).to(opt.device)
                pred_3D, mu, log_var = self.network_3D(target_3D, pred_2D_train)        
                #b_size = target_3D[0] #batch_size
                b_size = 8
                pred_3D = torch.reshape(pred_3D, (b_size, 17, 3))
                #print('pred_3D_check: ', pred_3D.shape) #[4, 16, 3]
                
                # print('pred_3D: ')
                # print(pred_3D.shape, pred_3D)

                # print('target_3D:') 
                # print(target_3D, target_3D.shape)

                #Original loss
                # criterion = nn.MSELoss()
                # loss_func = criterion(pred_3D, target_3D)

                # learning_loss_ = loss_func.clone().detach()
                # loss_func = loss_func.mean()
                #Original loss
                criterion = nn.MSELoss(reduction='none')
                loss_func = criterion(pred_3D, target_3D)
                learning_loss_ = torch.mean(loss_func, (1,2)).clone().detach()
                loss_func = loss_func.mean()
                #print('for loss', loss_func)
                
                
                #loss_func = criterion(pred_3D, target_3D[:, 0:16, :]).mean()
                #KL_loss = args.kl_factor * -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                KL_loss = self.conf.kl_factor*-0.5*torch.sum(1+log_var - mu.pow(2) - log_var.exp())
                total_loss = loss_func + KL_loss

                #LearnLoss
                #learning_loss_ = total_loss.clone().detach()


                #learning_loss_ = torch.mean(learning_loss_, dim=[1])
                if self.train_learning_loss:
                    loss_learnloss = self.learning_loss(hourglass_features, learning_loss_, self.learnloss_margin, input_2D, e) #input_2D
                    loss_learnloss.backward()
                    epoch_loss_learnloss.append(loss_learnloss.cpu().data.numpy())
                #LearnLoss

                rec = [float(total_loss.detach().cpu().numpy()), float(loss_func.detach().cpu().numpy()), float(KL_loss.detach().cpu().numpy())]

                self.optimizer.zero_grad() # 此行非固定 只要在loss.backward() 之前呼叫: 5-38
                total_loss.backward() # 反向傳遞
                self.optimizer.step() # 更新參數
                global_step+=1 


                total_loss_list.append(rec[0])
                loss_func_list.append(rec[1])
                kl_loss_list.append(rec[2])

                train_record["total_loss"].append(total_loss_list)
                train_record["loss_func"].append(loss_func_list)
                train_record["kl_loss"].append(kl_loss_list)
                
            epoch_loss = np.mean(train_record['total_loss'])
            self.tb_writer.add_scalar('Train', torch.Tensor([epoch_loss]), global_step)

            ##LearnLoss
            if self.train_learning_loss:
                epoch_loss_learnloss = np.mean(epoch_loss_learnloss)
            ##LearnLoss
            
            #torch.save(self.network_3D.state_dict(), self.model_save_path.format("model_epoch_{}.pth".format(e+1)))

            if best_val_hg >epoch_loss: #total_loss
                torch.save(self.network_3D.state_dict(), self.model_save_path.format("best_3D_model.pth"))
                # torch.save(self.learnloss_network.state_dict(),
                #            self.model_save_path.format("best_model_learnloss_hg.pth"))

                best_val_hg = epoch_loss #total_loss
                best_epoch_hg = e + 1

            # print("Loss at epoch {}/{}: (train) {}\t"
            #       "Learning Loss: (train) {}\t"
            #       "(validation: HG) {}\t"
            #       "(Validation: Learning Loss) {}\t"
            #       "(Best Model) {}".format(
            #     e+1,
            #     self.epoch,
            #     epoch_loss,
            #     epoch_loss_learnloss,
            #     validation_loss_hg,
            #     validation_learning_loss,
            #     best_epoch_hg))
            print("Loss at epoch {}/{}: \t"
                  "(Train: total loss) {}\t"
                  "(Best Model) {}".format(
                e+1,
                self.epochs_3D,
                epoch_loss,
                best_epoch_hg))
            
            loss_across_epochs.append(epoch_loss)
            f = open(self.model_save_path.format("loss_data.txt"), "w")
            f.write("\n".join([str(lsx) for lsx in loss_across_epochs]))
            f.close()

            if self.train_learning_loss:
                if best_val_learnloss > epoch_loss_learnloss:
                    torch.save(self.learnloss_network.state_dict(), self.model_save_path.format("best_ll_model.pth"))
                # torch.save(self.learnloss_network.state_dict(),
                #            self.model_save_path.format("best_model_learnloss_hg.pth"))

                    best_val_learnloss = epoch_loss_learnloss #total_loss
                    best_epoch_learnloss = e + 1
                print("Loss at epoch {}/{}: \t"
                  "(Train: learnloss loss) {}\t"
                  "(Best Model) {}".format(
                e+1,
                self.epochs_3D,
                epoch_loss_learnloss,
                best_epoch_learnloss))

        self.tb_writer.close()

    def learning_loss(self, hg_encodings, true_loss, margin, gt_per_img, epoch):
        '''
        Learning Loss module
        Refer:
        1. "Learning Loss For Active Learning, CVPR 2019"
        2. "A Mathematical Analysis of Learning Loss for Active Learning in Regression, CVPRW 2021"
        :param hg_encodings: (Dict of tensors) Intermediate (Hourglass) and penultimate layer output of the Hourglass network
        :param true_loss: (Tensor of shape [Batch Size]) Loss computed from HG prediction and ground truth
        :param margin: (scalar) tolerance margin between predicted losses
        :param gt_per_img: (Tensor, shape [Batch Size]) Number of ground truth per image
        :param epoch: (scalar) Epoch, used in learning loss warm start-up
        :return: (Torch scalar tensor) Learning Loss
        '''

        # Concatenate the layers instead of a linear combination
        with torch.no_grad():
            if self.learnloss_original:
                # hg_depth == 4 means depth is {1, 2, 3, 4}. If we want depth 5, range --> (1, 4+2)
                # encodings = torch.cat([hg_encodings[depth] for depth in range(1, self.hg_depth + 2)], dim=-1)
                encodings = hg_encodings['penultimate']

            else:
                # No longer concatenating, will now combine features through convolutional layers
                encodings = torch.cat([hg_encodings['feature_5'].reshape(self.batch_size, hg_encodings['feature_5'].shape[1], -1),               # 64 x 64
                                       hg_encodings['feature_4'].reshape(self.batch_size, hg_encodings['feature_4'].shape[1], -1),               # 32 x 32
                                       hg_encodings['feature_3'].reshape(self.batch_size, hg_encodings['feature_3'].shape[1], -1),               # 16 x 16
                                       hg_encodings['feature_2'].reshape(self.batch_size, hg_encodings['feature_2'].shape[1], -1),               # 8 x 8
                                       hg_encodings['feature_1'].reshape(self.batch_size, hg_encodings['feature_1'].shape[1], -1)], dim=2)       # 4 x 4

        emperical_loss, encodings = self.learnloss_network(encodings)
        emperical_loss = emperical_loss.squeeze()
        assert emperical_loss.shape == true_loss.shape, "Mismatch in Batch size for true and emperical loss"

        with torch.no_grad():
            # Scale the images as per the number of joints
            # To prevent DivideByZero. PyTorch does not throw an exception to DivideByZero
            gt_per_img = torch.sum(gt_per_img, dim=1)
            #print('gt_per_img: ', gt_per_img.shape) #[4, 256, 3])
            gt_per_img = torch.mean(gt_per_img, (1,2))
            gt_per_img += 0.1
            if self.learnloss_obj == 'prob':
                # print('Error', true_loss.shape) #[4]
                # print('Error2: ', gt_per_img.shape) #[4, 256, 3]

                true_loss = true_loss / gt_per_img

            # Splitting into pairs: (i, i+half)
            half_split = true_loss.shape[0] // 2

            true_loss_i = true_loss[: half_split]
            true_loss_j = true_loss[half_split: 2 * half_split]

        emp_loss_i = emperical_loss[: (emperical_loss.shape[0] // 2)]
        emp_loss_j = emperical_loss[(emperical_loss.shape[0] // 2): 2 * (emperical_loss.shape[0] // 2)]

        # Loss according to CVPR '19
        if self.learnloss_obj == 'pair':
            loss_sign = torch.sign(true_loss_i - true_loss_j)
            loss_emp = (emp_loss_i - emp_loss_j)

            # Learning Loss objective
            llal_loss = torch.max(torch.zeros(half_split, device=loss_sign.device), (-1 * (loss_sign * loss_emp)) + margin)

        # Loss according to CVPR '21
        elif self.learnloss_obj == 'prob':
            with torch.no_grad():
                true_loss_ = torch.cat([true_loss_i.reshape(-1, 1), true_loss_j.reshape(-1, 1)], dim=1)
                true_loss_scaled = true_loss_ / torch.sum(true_loss_, dim=1, keepdim=True)

            emp_loss_ = torch.cat([emp_loss_i.reshape(-1, 1), emp_loss_j.reshape(-1, 1)], dim=1)
            emp_loss_logsftmx = torch.nn.LogSoftmax(dim=1)(emp_loss_)
            llal_loss = torch.nn.KLDivLoss(reduction='batchmean')(input=emp_loss_logsftmx, target=true_loss_scaled)

        else:
            raise NotImplementedError('Currently only "pair" or "prob" supported. ')

        if self.learnloss_warmup <= epoch:
            return torch.mean(llal_loss)
        else:
            return 0.0 * torch.mean(llal_loss)

    def validation_3D(self, e):
        global_step_val =0
        with torch.no_grad():
            epoch_val_hg = []

            if self.train_learning_loss:
                epoch_val_learnloss = []
            
            self.network_2D.eval()
            self.network_3D.eval()
            if self.train_learning_loss:
                self.learnloss_network.eval()
            
            total_loss_val_list, loss_func_val_list, kl_loss_val_list = [], [], [] # Calvin
            train_record_val = {"total_loss":[], "loss_func":[], "kl_loss": []} # Calvin

            print('Validation for epoch: {}'.format(e+1))
            for i, batch in enumerate(tqdm(self.val_dataset_obj)):
                input_val, target_val, meta_val = batch['input'], batch['target'], batch['meta']
                input_val = input_val.to(non_blocking=True, device=self.cuda_devices[-1])
                target_val = meta_val['gt_3d'].cuda(device=self.opt.device, non_blocking=True)
                input_val = input_val.permute(0, 2, 3, 1) 
                outputs_val, hourglass_features_val = self.network_2D(input_val)

                outputs_val = outputs_val.cpu().detach().numpy().mean(axis= 1)
                outputs_val = self.get_preds(outputs_val) #(B, 16, 2)
                outputs_val = outputs_val.astype(np.float32)
                
                outputs_val_pred = []
                for i in range(outputs_val.shape[0]):
                    b_torse = (outputs_val[i][2]+outputs_val[i][3])/2
                    outputs_val_pred.append(np.concatenate((outputs_val[i], [b_torse]), axis=0)[self.idx_2D_to_3D])
                
                outputs_val_pred = torch.tensor(outputs_val_pred).to(opt.device)
                outputs_val_3D, mu_val, log_var_val = self.network_3D(target_val, outputs_val_pred)  

                b_size = 4
                pred_3D_val = torch.reshape(outputs_val_3D, (b_size, 17, 3))

                criterion = nn.MSELoss()
                loss_func_val = criterion(pred_3D_val, target_val).mean()
                KL_loss_val = self.conf.kl_factor*-0.5*torch.sum(1+log_var_val - mu_val.pow(2) - log_var_val.exp())
                total_loss_val = KL_loss_val+loss_func_val
                rec_val = [float(total_loss_val.detach().cpu().numpy()), float(loss_func_val.detach().cpu().numpy()), float(KL_loss_val.detach().cpu().numpy())]

                total_loss_val_list.append(rec_val[0])
                loss_func_val_list.append(rec_val[1])
                kl_loss_val_list.append(rec_val[2])

            epoch_loss_val = np.mean(train_record_val['total_loss'])
            self.tb_writer.add_scalar('Validation', torch.Tensor([epoch_loss_val]), global_step_val)
            global_step_val+=1

            """
            if best_val_hg >epoch_loss: #total_loss
                torch.save(self.network_3D.state_dict(), self.model_save_path.format("best_3D_model.pth"))
                # torch.save(self.learnloss_network.state_dict(),
                #            self.model_save_path.format("best_model_learnloss_hg.pth"))

                best_val_hg = epoch_loss #total_loss
                best_epoch_hg = e + 1

            # print("Loss at epoch {}/{}: (train) {}\t"
            #       "Learning Loss: (train) {}\t"
            #       "(validation: HG) {}\t"
            #       "(Validation: Learning Loss) {}\t"
            #       "(Best Model) {}".format(
            #     e+1,
            #     self.epoch,
            #     epoch_loss,
            #     epoch_loss_learnloss,
            #     validation_loss_hg,
            #     validation_learning_loss,
            #     best_epoch_hg))
            print("Loss at epoch {}/{}: \t"
                  "(Train: total loss) {}\t"
                  "(Best Model) {}".format(
                e+1,
                self.epochs_3D,
                epoch_loss_val,
                best_epoch_hg))
            
            loss_across_epochs.append(epoch_loss)
            f = open(self.model_save_path.format("validation_data.txt"), "w")
            f.write("\n".join([str(lsx) for lsx in loss_across_epochs]))
            f.close()
        self.tb_writer.close()
        """
                

class Test_3D(object):
    '''
    Inference 3D model performance
    '''
    def __init__(self, network, dataset_obj, conf, opt):
        self.network = network # load 2D model
        self.dataset_obj = dataset_obj
        self.conf = conf
        self.opt = opt

        self.ind_to_jnt = {0: 'rfoot', 1: 'rknee', 2: 'rhip', 3: 'lhip', 4: 'lknee', 5: 'lfoot', 6: 'torso', 7: 'chest',
            8: 'Neck', 9: 'Head', 10: 'rhand', 11: 'relb', 12: 'rsho', 13:'lsho', 14: 'lelb', 15:'lhand'}
        self.model_save_path = conf.model_save_path
        self.idx_2D_to_3D = [16, 3, 4, 5, 2, 1, 0, 7, 6, 8, 9, 12, 11, 10, 13, 14, 15]

        self.device = opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
        # in main?
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.network_3D = cVAE(self.encoder, self.decoder).to(self.device)
        
        print('Load 3D model from:', conf.model_load_path_3D)
        self.network_3D.load_state_dict(torch.load(os.path.join(conf.model_load_path_3D, 'model_checkpoints/best_3D_model.pth')))
        
        print('Done with loading the 3D model')

    def get_preds(self, hm, return_conf=False):
        assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
        h = hm.shape[2]
        w = hm.shape[3]
        hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
        idx = np.argmax(hm, axis = 2)
        
        preds = np.zeros((hm.shape[0], hm.shape[1], 2))
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                preds[i, j, 0], preds[i, j, 1] = idx[i, j] % w, idx[i, j] / w
        if return_conf:
            conf = np.amax(hm, axis = 2).reshape(hm.shape[0], hm.shape[1], 1)
            return preds, conf
        else:
            return preds # shape: [batch, joint, 0:1]

    def absolute_to_root_relative(joints, root_index):
        root = joints.narrow(-2, root_index, 1)
        return joints - root

    def mpjpe(predicted, target):
        """
        Mean per-joint position error (i.e. mean Euclidean distance),
        often referred to as "Protocol #1" in many papers.
        """
        assert predicted.shape == target.shape
        predicted = absolute_to_root_relative(predicted, 0)
        target = absolute_to_root_relative(target, 0)

        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

    def inference(self):
        self.network.eval()
        self.network_3D.eval()
        image_index = torch.tensor([]) 
        image_nor = torch.tensor([]) 
        output_xy = np.empty((1, 16, 2))
        target_xy = np.empty((1, 16, 2))
        mpjpe_record = []
        max_batch=10000
        c=0
        for i, batch in enumerate(tqdm(self.dataset_obj)):
            with torch.no_grad():
                input_val, target_val, meta_val = batch['input'], batch['target'], batch['meta']
                input_val = input_val.to(non_blocking=True, device=opt.device)
                target_3D = meta_val['gt_3d'].cuda(device=self.opt.device, non_blocking=True)
                
                input_val = input_val.permute(0, 2, 3, 1) 
                outputs_val, hourglass_features_val = self.network(input_val) # 算出來的heatmap
                #target_3D = target_3D[:, 1:17, :]

                outputs_val2 = outputs_val.cpu().detach().numpy().copy()
                outputs_val2 = outputs_val.mean(axis= 1)

                target_val = target_val.to(non_blocking=True, device=opt.device)# check 2D performance, can delete later
                target_val2 = self.get_preds(target_val.detach().cpu().numpy()) # check 2D performance, delete later
                pred_val = self.get_preds(outputs_val2.detach().cpu().numpy()) #(1, 16, 2)
                pred_val = pred_val.astype(np.float32)

                #add and switch
                outputs_val_pred = []
                
                for i in range(pred_val.shape[0]):
                    b_torse = (pred_val[i][2]+pred_val[i][3])/2
                    outputs_val_pred.append(np.concatenate((pred_val[i], [b_torse]), axis=0)[self.idx_2D_to_3D])
                c+=1
                
                #outputs_val_pred = torch.tensor(outputs_val_pred).to(opt.device)
                #outputs_val_3D, mu_val, log_var_val = self.network_3D(target_val, outputs_val_pred)  

                #b_size = 4
                #pred_3D_val = torch.reshape(outputs_val_3D, (b_size, 17, 3))

                # add and switch


                        
                pred_3D, mu, log_var = self.network_3D.forward(torch.tensor(target_3D, device=opt.device), torch.tensor(outputs_val_pred, device=opt.device))
                b_size = 1
                pred_3D = torch.reshape(pred_3D, (b_size, 17, 3))
                
                if c==1900:
                    print('index')
                    print(meta_val['index'])
                    # print('2D-GT')
                    # print(target_val2, target_val2.shape)
                    # print('2D-pred')
                    # print(pred_val, pred_val.shape)

                    # print('Modify 2D-pred')
                    # pred_val = torch.tensor(pred_val, device=opt.device)
                    # print(pred_val, pred_val.shape)
                    
                    print('3D-GT')
                    print(target_3D, target_3D.shape)
                    print('3D-Pred')
                    print(pred_3D)

                
                mpjpe_rec = mpjpe(pred_3D, target_3D).item()
                mpjpe_record.append(mpjpe_rec)
            
            fill_size = len(str(max_batch))
            print("\rProgress: "+str(i+1).zfill(fill_size)+"/"+str(max_batch), end="")
            
        print("\nDone~~")
        mpjpe_record = np.array(mpjpe_record)

        mpjpe_mean = mpjpe_record.mean()
        mpjpe_std = mpjpe_record.std()

        return {"mpjpe":[float(mpjpe_mean), float(mpjpe_std)],
            # "n_mpjpe": [float(n_mpjpe_mean), float(n_mpjpe__std)]
            }

class Pick_3D(object):
    '''
    Inference 3D model performance
    '''
    def __init__(self, network, dataset_obj, conf, opt):
        self.network = network # load 2D model
        #self.dataset_obj = dataset_obj
        self.conf = conf
        self.opt = opt
        self.torch_dataloader = dataset_obj

        self.ind_to_jnt = {0: 'rfoot', 1: 'rknee', 2: 'rhip', 3: 'lhip', 4: 'lknee', 5: 'lfoot', 6: 'torso', 7: 'chest',
            8: 'Neck', 9: 'Head', 10: 'rhand', 11: 'relb', 12: 'rsho', 13:'lsho', 14: 'lelb', 15:'lhand'}
        self.model_save_path = conf.model_save_path
        self.idx_2D_to_3D = [16, 3, 4, 5, 2, 1, 0, 7, 6, 8, 9, 12, 11, 10, 13, 14, 15]

        self.device = opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
        # in main?
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.network_3D = cVAE(self.encoder, self.decoder).to(self.device)
        
        print('Load 3D model from:', conf.model_load_path_3D)
        self.network_3D.load_state_dict(torch.load(os.path.join(conf.model_load_path_3D, 'model_checkpoints/best_3D_model.pth')))
        
        print('Done with loading the 3D model')

    def get_preds(self, hm, return_conf=False):
        assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
        h = hm.shape[2]
        w = hm.shape[3]
        hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
        idx = np.argmax(hm, axis = 2)
        
        preds = np.zeros((hm.shape[0], hm.shape[1], 2))
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                preds[i, j, 0], preds[i, j, 1] = idx[i, j] % w, idx[i, j] / w
        if return_conf:
            conf = np.amax(hm, axis = 2).reshape(hm.shape[0], hm.shape[1], 1)
            return preds, conf
        else:
            return preds # shape: [batch, joint, 0:1]

    def absolute_to_root_relative(joints, root_index):
        root = joints.narrow(-2, root_index, 1)
        return joints - root

    def mpjpe(predicted, target):
        """
        Mean per-joint position error (i.e. mean Euclidean distance),
        often referred to as "Protocol #1" in many papers.
        """
        assert predicted.shape == target.shape
        predicted = absolute_to_root_relative(predicted, 0)
        target = absolute_to_root_relative(target, 0)

        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

    def inference(self):
        self.network.eval()
        self.network_3D.eval()
        image_index = torch.tensor([]) 
        image_nor = torch.tensor([]) 
        output_xy = np.empty((1, 16, 2))
        target_xy = np.empty((1, 16, 2))
        mpjpe_record = []
        max_batch=10000
        c=0
                    # train_record = {"total_loss":[], "loss_func":[], "kl_loss": []} # Calvin

        for (inp, out, meta, images) in tqdm(self.torch_dataloader):
            '''
            meta = {'index' : self.idxs[index], 'center' : c, 'scale' : s, 
                    'gt_3d': gt_3d, 'pts_crop': pts_crop, 'normalizer':normalizer}

            {'input': inp, 'target': out, 'meta': meta, 
                    'reg_target': reg_target, 'reg_ind': reg_ind, 'reg_mask': reg_mask, 'image':inp_copy}#, 'ht':out}
            '''
            
            input_2D = inp.permute(0, 2, 3, 1)
            input_2D = input_2D.cuda(device=self.opt.device, non_blocking=True)
            target_3D = meta['gt_3d'].cuda(device=self.opt.device, non_blocking=True) #x #meta['gt_3d']
            pred_2D, hourglass_features = self.network(input_2D) #c

            '''
            input_2D:  <class 'torch.Tensor'> torch.Size([B, 256, 256, 3])
            target_3D:  <class 'torch.Tensor'> torch.Size([B, 17, 3])
            pred_2D: <class 'torch.Tensor'> torch.Size([B, 2, 16, 64, 64])
            pred_2D_max: torch.Size([B, 16, 2]) torch.float32
            target_3D: torch.Size([12, 16, 3]) torch.float32
            pred_3D: torch.Size([12, 48])
            '''
            pred_2D = pred_2D.cpu().detach().numpy().mean(axis= 1)
            pred_2D_max = self.get_preds(pred_2D) #(B, 16, 2)
            pred_2D_max = pred_2D_max.astype(np.float32)
            
            pred_2D_train = []
            # 加入element and index
            for i in range(pred_2D_max.shape[0]):
                b_torse = (pred_2D_max[i][2]+ pred_2D_max[i][3])/2
                pred_2D_train.append(np.concatenate((pred_2D_max[i], [b_torse]), axis=0)[self.idx_2D_to_3D])
            pred_2D_train = np.array(pred_2D_train)
            # 加入element and index

            pred_2D_train = torch.tensor(pred_2D_train).to(opt.device)
            pred_3D, mu, log_var = self.network_3D(target_3D, pred_2D_train)        
            #b_size = target_3D[0] #batch_size
            b_size = 1
            pred_3D = torch.reshape(pred_3D, (b_size, 17, 3))
        
        # for i, batch in enumerate(tqdm(self.dataset_obj)):
        #     with torch.no_grad():
        #         input_val, target_val, meta_val = batch['input'], batch['target'], batch['meta']
        #         input_val = input_val.to(non_blocking=True, device=opt.device)
        #         target_3D = meta_val['gt_3d'].cuda(device=self.opt.device, non_blocking=True)
                
        #         input_val = input_val.permute(0, 2, 3, 1) 
        #         outputs_val, hourglass_features_val = self.network(input_val) # 算出來的heatmap
        #         #target_3D = target_3D[:, 1:17, :]

        #         outputs_val2 = outputs_val.cpu().detach().numpy().copy()
        #         outputs_val2 = outputs_val.mean(axis= 1)

        #         target_val = target_val.to(non_blocking=True, device=opt.device)# check 2D performance, can delete later
        #         target_val2 = self.get_preds(target_val.detach().cpu().numpy()) # check 2D performance, delete later
        #         pred_val = self.get_preds(outputs_val2.detach().cpu().numpy()) #(1, 16, 2)
        #         pred_val = pred_val.astype(np.float32)

        #         #add and switch
        #         outputs_val_pred = []
                
        #         for i in range(pred_val.shape[0]):
        #             b_torse = (pred_val[i][2]+pred_val[i][3])/2
        #             outputs_val_pred.append(np.concatenate((pred_val[i], [b_torse]), axis=0)[self.idx_2D_to_3D])
        
            mpjpe_rec = mpjpe(pred_3D, target_3D).item()
            mpjpe_record.append(mpjpe_rec)
        
        fill_size = len(str(max_batch))
        print("\rProgress: "+str(i+1).zfill(fill_size)+"/"+str(max_batch), end="")
            
        print("\nDone~~")
        mpjpe_record = np.array(mpjpe_record)

        mpjpe_mean = mpjpe_record.mean()
        mpjpe_std = mpjpe_record.std()

        return {"mpjpe":[float(mpjpe_mean), float(mpjpe_std)],
            # "n_mpjpe": [float(n_mpjpe_mean), float(n_mpjpe__std)]
            }

    

from opts import opts
# utils_my 有改動(github)
def main(opt,conf):
    # 把GT跟heatmap plot出來
    print('Load info')
    args = conf.args # get config_3D之後要打開->用在2D部分
    args_3D = conf.args_3D # args_3D['loss_type'] # MSE

    opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))

    # model place here
    hg_param = args['hourglass']

    print('Load the model from: ', conf.model_load_path)
    model, ll = load_models(conf=conf, load_hg=conf.model_load_hg, load_learnloss=conf.model_load_learnloss,
                                     best_model=conf.best_model, hg_param=hg_param, model_dir=conf.model_load_path)
   
    hyperparameters = define_hyperparams(conf, model, ll)
    writer = SummaryWriter(log_dir=os.path.join(conf.model_save_path[:-20], 'tensorboard'))

    ## AL 
    active_learning_obj = ActiveLearning(conf=conf,hg_network=model, learnloss_network=ll)

    # Check
    #h36m = H36M(opt, 'train', activelearning_obj=active_learning_obj)
    #打開(conf.demo再註解)
    if conf.demo or conf.metric_3D or conf.pick_3D:
        pass
    else:
        h36m, h36m_2 = load_h36m(opt, 'train', precached_h36m = conf.precached_h36m)
        #print('main_h36m: ', h36m.keys()) #(['input', 'target', 'meta', 'reg_target', 'reg_ind', 'reg_mask', 'image'])
        h36m_Data = H36M_AL(h36m_dict = h36m, h36m_dict_2 =h36m_2, activelearning_obj = active_learning_obj, getitem_dump =conf.model_save_path, conf = conf, **args)
        print("Done with active learning dataset")

        #0123 real batch size for 3D model hide here, need to modify the para as vars
        torch_dataloader = torch.utils.data.DataLoader(h36m_Data , batch_size=8,
                                                                shuffle=True, num_workers=0, drop_last=True)

        print("Give the memory back!")
        del h36m
        del h36m_2
        print("Fininsh delelte!")

    # #打開

    val_loader = torch.utils.data.DataLoader(H36M(opt, 'val'), 
    batch_size=1, 
    shuffle=False,
    num_workers=0,
    pin_memory=True)

    # load 3D model
    encoder = Encoder()
    decoder = Decoder()
    model_3D =  cVAE(encoder, decoder).to(opt.device)
    hyperparameter_3D = define_hyperparams_3D(conf, model_3D, ll)
    
# pick
    if conf.train_3D:
        print('Start training 3D network-------')
        # self, network_2D, network_3D, learnloss_network, hyperparameters, dataset_obj, val_dataset_obj, tb_writer, conf, opt):
        train_3D_obj = Train_3D(network_2D = model, network_3D = model_3D, learnloss_network=ll, hyperparameters=hyperparameter_3D,
                            dataset_obj=torch_dataloader, conf=conf, tb_writer=writer, opt= opt, val_dataset_obj=val_loader)
        train_3D_obj.train_model_3D()
        print('Done with training 3D model')

    if conf.metric_3D:
        print('Start inference 3D network-------')
        metric_3D_obj = Test_3D(network = model,dataset_obj=val_loader, conf = conf, opt= opt)
        eval_results_test = metric_3D_obj.inference()
        print("MPJPE: {}, MPJPE_STD: {}".format(eval_results_test['mpjpe'][0], eval_results_test['mpjpe'][1]))

        exit()

    if conf.pick_3D:
        print('Start inference the pick image 3D:')
        annotated_pick_pre_3D = np.load(os.path.join(conf.model_load_pre_3D, 'model_checkpoints/annotation.npy'))
        annotated_pick_now_3D = np.load(os.path.join(conf.model_load_now_3D, 'model_checkpoints/annotation.npy'))
        unlabelled_idx = np.array(list(set(annotated_pick_now_3D)-set(annotated_pick_pre_3D)))

        h36m, h36m_2 = load_h36m(opt, 'train', precached_h36m = conf.precached_h36m)
        for k in h36m.keys():
            for i in h36m_2[k]:
                h36m[k].append(i)
        h36m_pick = H36M_pick(h36m_dict = h36m, conf =conf, indices = unlabelled_idx)

        torch_dataloader_pick = torch.utils.data.DataLoader(h36m_pick , batch_size=1,
                                                            shuffle=True, num_workers=0, drop_last=True)
                
        #metric_pick_obj = Test3(network = model_pre, dataset_obj = torch_dataloader_pick, conf = conf)
        metric_pick_obj = Pick_3D(network = model, dataset_obj = torch_dataloader_pick, conf = conf, opt=opt)
        eval_result_pick = metric_pick_obj.inference()
        
        print("Finsh pick indices")
        print("MPJPE: {}, MPJPE_STD: {}".format(eval_result_pick['mpjpe'][0], eval_result_pick['mpjpe'][1]))
        exit()
    

    if conf.pick:
        print('Start inference the pick image:')
        annotated_pick_pre = np.load(os.path.join(conf.model_load_pre, 'model_checkpoints/annotation.npy'))
        annotated_pick_now = np.load(os.path.join(conf.model_load_now, 'model_checkpoints/annotation.npy'))
        unlabelled_idx = np.array(list(set(annotated_pick_now)-set(annotated_pick_pre)))
        
        h36m_pick = H36M_pick(h36m_dict = h36m, conf =conf, indices = unlabelled_idx)
        torch_dataloader_pick = torch.utils.data.DataLoader(h36m_pick , batch_size=1,
                                                            shuffle=True, num_workers=0, drop_last=True)


        model_pre, ll_pre = load_models(conf=conf, load_hg=conf.model_load_hg, load_learnloss=conf.model_load_learnloss,
                                     best_model=conf.best_model, hg_param=hg_param, model_dir=conf.model_load_pre)
        
        metric_pick_obj = Test3(network = model_pre, dataset_obj = torch_dataloader_pick, conf = conf)
        #metric_pick_obj.inference()
        metric_pick_obj.eval()
        print("Finsh pick indices")

        # print('check: ', len(unlabelled_idx))
        exit()



    if conf.train:
        print('Start training----')
        train_obj = Train2(network=model, learnloss=ll, hyperparameters=hyperparameters,
                            dataset_obj=torch_dataloader, conf=conf, tb_writer=writer, opt= opt, val_dataloader=val_loader)
        train_obj.train_model()
        print('Done with training----')

    # for (inp, out, meta, image) in tqdm(torch_dataloader):
    #     print(type(inp), inp.shape) #class 'torch.Tensor'> torch.Size([12, 3, 256, 256])
    #     print(type(out), out.shape)# <class 'torch.Tensor'> torch.Size([12, 16, 64, 64]

    #     exit()

    # train_loader = torch.utils.data.DataLoader(h36m_Data, 
    # batch_size=opt.batch_size * len(opt.gpus), 
    # shuffle=True, # if opt.debug == 0 else False,
    # num_workers=opt.num_workers,
    # pin_memory=True)

    # print("Try the dataloader")

    
    # for i, batch in enumerate(train_loader):
    #      input, target, meta= batch['input'], batch['target'], batch['meta']
    #      print('input', type(input), input.shape)
    #      exit(False)

    '''
    # Load Data
    train_loader = torch.utils.data.DataLoader(H36M(opt, 'train', activelearning_obj=active_learning_obj), 
    batch_size=opt.batch_size * len(opt.gpus), 
    shuffle=True, # if opt.debug == 0 else False,
    num_workers=opt.num_workers,
    pin_memory=True)


    val_loader = torch.utils.data.DataLoader(H36M(opt, 'val', activelearning_obj=active_learning_obj), 
    batch_size=1, 
    shuffle=False,
    num_workers=0,
    pin_memory=True)
    '''

    # for i, batch in enumerate(val_loader):
    #     #data_time.update(time.time() - end)
    #     '''
    #     input: torch.Size([32, 3, 256, 256])
    #     target: torch.Size([32, 16, 64, 64])

    #     '''
    #     input, target, meta= batch['input'], batch['target'], batch['meta']
    #     #print(meta['index'])
    #     vis_merge(input, meta, target, meta['index'])
        
    #     # vis_try_2(input, meta)
    #     # vis_joint_name(target, meta['index'])
        
        
        
       
       
    #     # vis_try_ht(target)
        
    #     #畫出heatmap
    #     # print(info[0][0])
    #     # print('index: ', meta['index'])
    #     # b =torch.zeros((64, 64))
    #     # for i in range(15):
    #     #     b+=info[0][i]

    #     # plt.imshow(b, cmap='magma')
    #     # plt.savefig("mygraph2.png")
        
    #     #畫出keypoints
    #     # print(i)
    #     # print(input.shape)
    #     # print(target.shape)
    #     # print(meta.keys()) # dict_keys(['index', 'center', 'scale', 'gt_3d', 'pts_crop'])
    #     # print(input[0])
    #     # print(meta['gt_3d'][0])
        
    #     # # for i in meta['gt_3d'][0]: 
    #     # #     print(i[:2])
    #     # import matplotlib.pyplot as plt
    #     # print(meta['index'][0])
    #     # for i in meta['gt_3d'][0]:
    #     #     plt.scatter(i[0], i[1])

    #     # plt.savefig("mygraph.png")
    
    
    if conf.train:
        print('Start training----')
        train_obj = Train(network=model, learnloss=ll, hyperparameters=hyperparameters,
                            dataset_obj=train_loader, conf=conf, tb_writer=writer, opt= opt, val_dataloader=val_loader)
        train_obj.train_model()
        print('Done with training----')

    # TODO 每個epoch vis 一下training結果

    # Sample data AL


    # def vis_try_ht_pred_2(info, image):


    if conf.metric:
        print('Start Inferencing----')
        metric_obj = Test2(network=model, dataset_obj=val_loader, conf=conf)
        metric_obj.eval()
        # model_infer = metric_obj.inference()
        # gt_csv, pred_csv = metric_obj.keypoint(model_infer)
        #metric_obj.inference()
        #metric_obj.eval()
        print("Done with metric----")
        # network_inf, learnloss_inf = load_models(conf=conf, load_hg=True, load_learnloss=False, best_model=True,
        #                                 hg_param=hg_param, model_dir=conf.model_save_path[:-20])


    if conf.demo:
        image_ext = ['jpg', 'jpeg', 'png']
        mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
        def is_image(file_name):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            return ext in image_ext
        
        def get_preds(hm, return_conf=False):
            assert len(hm.shape) == 4, 'Input must be a 4-D tensor'

            h = hm.shape[2]
            w = hm.shape[3]
            hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
            idx = np.argmax(hm, axis = 2)
            
            preds = np.zeros((hm.shape[0], hm.shape[1], 2))
            for i in range(hm.shape[0]):
                for j in range(hm.shape[1]):
                    preds[i, j, 0], preds[i, j, 1] = idx[i, j] % w, idx[i, j] / w
            if return_conf:
                conf = np.amax(hm, axis = 2).reshape(hm.shape[0], hm.shape[1], 1)
                return preds, conf
            else:
                return preds # shape: [batch, joint, 0:1]

        def arg_max(img):
            '''
            Find the indices corresponding to maximum values in the heatmap
            :param img: (Numpy array of size=64x64) Heatmap
            :return: (Torch tensor of size=2) argmax of the image
            '''
            img = torch.tensor(img)
            assert img.dim() == 2, 'Expected img.dim() == 2, got {}'.format(img.dim())

            h = img.shape[0]
            w = img.shape[1]
            rawmaxidx = img.flatten().argmax()

            max_u = int(rawmaxidx) // int(w)
            max_v = int(rawmaxidx) % int(w)

            return torch.FloatTensor([max_u, max_v])

        def vis_demo_merge(info, target, filename):
            mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
            print(info[0].shape) 
            input2 = info[0]
            #input2 = info[0].permute(1, 2, 0) 
            input2 = input2.cpu().numpy()
            input2 = (input2*std + mean) # image 的matric

            input2_array = (input2*255).astype(np.uint8)
            #print(input2_array.dtype)
            #cv2.imshow('image',input2_array)
            #cv2.imwrite('../debug_vis_1/image_pred_img.png', input2_array) # RGB
            cv2.imwrite('../debug_vis/h36m_100_model_folder/image_pred_img.png', input2_array) # RGB

            image_to_write = cv2.cvtColor(input2_array, cv2.COLOR_RGB2BGR) #cv2.COLOR_BGR2RGB
            cv2.imwrite('../debug_vis/h36m_100_model_folder/image_pred_cvt.png', image_to_write) # RGB
            bottom_pic = input2_array

            b =torch.zeros((64, 64))
            for i in range(16): 
                b=target[0][i]

                b = b.detach().cpu().numpy()
                b_array = (b*255).astype(np.uint8)
                #print(b_array.dtype)
                heatmap = cv2.applyColorMap(b_array, cv2.COLORMAP_HOT)
                top_pic = cv2.resize(heatmap, (256, 256), interpolation = cv2.INTER_AREA)
                print(top_pic.shape)
                print(bottom_pic.shape)
                overlapping_pic = cv2.addWeighted(bottom_pic, 0.6, top_pic, 0.4, 0) #(256, 256, 3) and unit8
                cv2.imwrite('../debug_vis/h36m_100_model_folder/image_{}merge_pred_joints_{}.png'.format(filename, i), overlapping_pic)
                
                #plt.savefig("../debug_vis/joints/image_{}_joint_{}.png".format(file_name, i))

            return print("Merge!")

        def vis_2D_new(pred, filename):
            ind_to_jnt_new = ['rfoot',
                'rknee',
                'rhip',
                'lhip',
                'lknee',
                'lfoot',
                'torso',
                'chest',
                'Neck',
                'Head',
                'rhand',
                'relb',
                'rsho',
                'lsho',
                'lelb']
            fig, ax = plt.subplots()
            for i in range(15):
                ax.scatter(pred[i][0], pred[i][1])
                
            for i, txt in enumerate(ind_to_jnt_new):
                ax.annotate(txt, (pred[i][0], pred[i][1]))
                
                #plt.text(pred[i][0]*(1+0.01), pred[i][1]*(1+0.01), ind_to_jnt[i], fontsize=8)
            plt.savefig('2D_scatter_{}.png'.format(filename))

        def vis_2D(pred_info, filename):
            for i in range(15):
                plt.scatter(pred_info[i][0], pred_info[i][1])
                plt.text(pred_info[i][0]*(1+0.01), pred_info[i][1]*(1+0.01), ind_to_jnt[i], fontsize=10)
            return plt.savefig('2D_scatter_{}.png'.format(filename))

        def demo_image(image, model, opt, file_name):

            '''
            image: (np.array) (1000, 1000, 3), dtype = unit8

            inp: (np.array) (256, 256, 3), dtype = unit8
            '''
            # cv2.imshow('Het',image)
            # cv2.waitKey(0)
            # exit()
            s = max(image.shape[0], image.shape[1]) * 1.0
            c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
            trans_input = get_affine_transform(
                c, s, 0, [opt.input_w, opt.input_h])
            inp = cv2.warpAffine(image, trans_input, (opt.input_w, opt.input_h),
                                    flags=cv2.INTER_LINEAR)
            inp2 = inp.copy()
            # cv2.imshow('Het',inp)
            # cv2.waitKey(0)
            # exit()

            inp = (inp / 255. - mean) / std
            inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            inp = torch.from_numpy(inp).to(opt.device)
            inp = inp.permute(0, 2, 3, 1) # 加了這行

            #print(inp.shape, type(inp), inp.dtype)
            #inp_show = inp.clone().cpu().numpy()
            # cv2.imshow('Het',inp_show[0].astype(np.uint8))
            # cv2.waitKey(0)
            # exit()
            #print('Inference test',inp.shape) #torch.Size([1, 256, 256, 3])
            #out = model(inp)[-1]
            out, _ = model(inp)
            
            inp_vis = inp.clone()
            #print(out.shape) # ([1, 2, 16, 64, 64])
            #print(out[:,-2].shape)# ([1, 16, 64, 64])
            #exit()
            '''
            這邊是try to 找output 的x, y 座標
            outputs = out[:, -2].detach()
            a = []
            for i in range(16):
                b=0
                b= arg_max(outputs[0][i])
                a.append(b)
            '''
            #print(outputs.shape) #torch.Size([1, 16, 64, 64])
            
            #vis_try_ht_pred(outputs) #成功視覺化
            #pred = get_preds(out[:,-2].detach().cpu().numpy()) #(1, 16, 2)
            pred = get_preds(out[:,-2].detach().cpu().numpy())[0] #(joint, 0:1) (16, 2)
            #print(pred.shape) #(16, 2)
            #pred = get_preds(out['hm'].detach().cpu().numpy())[0]

            out_vis = out.clone()
            out_vis = out_vis.mean(axis=1)


            out2 = out.cpu().detach().numpy().copy()
            out2 = out2.mean(axis= 1)
            
            print(inp_vis.shape) #torch.Size([1, 256, 256, 3])
            print(inp_vis.dtype)
            print(out_vis.shape) #
            print(out_vis.dtype)


            inp_show = inp_vis.detach().cpu().numpy()[0]
            cv2.imshow('Input image',inp_show.astype(np.uint8))
            cv2.waitKey(0)

            vis_demo_merge(inp_vis, out_vis, file_name)
            
            #vis_try_ht_pred_2(out2, file_name) # out2 有16張 64x64 heatmap
            pred = transform_preds(pred, c, s, (opt.output_w, opt.output_h)) 
            
            '''
            pred(after transform): (np.array) (16, 2)
            c: [500 500]
            s: 1000.0
            opt.output_w: (int) 64
            opt.output_h: (int) 64 
            '''
            # pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(), 
            #                     out['depth'].detach().cpu().numpy())[0]
            
            # vis_2D_new(pred, file_name)
            # print('pred: ', type(pred), pred, pred.shape, pred.dtype)
            

            debugger = Debugger()
            debugger.add_img(image)
            debugger.add_point_2d(pred, (255, 150, 0))
            #debugger.add_point_2d_2(pred, (114, 128, 250), h36m['target'][0], (250, 128, 250)) #RGB BGR 
            
            #debugger.add_point_3d(pred_3d, 'b')
            debugger.show_all_imgs(pause=True)
            #ebugger.show_3d()



        # opt.heads['depth'] = opt.num_output
        # if opt.load_model == '':
        #     opt.load_model = '../Experiments/Pre/fusion_3d_var/pth'#'../Experiments/Pre/fusion_3d_var/pth' # '../Experiments/1012_Test_713/model_checkpoints/best_model.pth'
        # if opt.gpus[0] >= 0:
        #     opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
        # else:
        #     opt.device = torch.device('cpu')

        model = model.to(opt.device)
        model.eval()

        if os.path.isdir(opt.demo):
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                if is_image(file_name):
                    image_name = os.path.join(opt.demo, file_name)
                    print('Running {} ...'.format(image_name))
                    image = cv2.imread(image_name)
                    demo_image(image, model, opt, file_name)
        elif is_image(opt.demo):
            print('Running {} ...'.format(opt.demo))
            image = cv2.imread(opt.demo)
            demo_image(image, model, opt)

        def vis_try_ht_pred(info, index):
            # heatmap # info放target
            b =torch.zeros((64, 64))
            for i in range(15): 
                b=info[i]
                
                plt.imshow(b, cmap='magma')
                plt.savefig("../1115_debug_vis_pred/Training_pred_heatmap_{}_{}.png".format(index, i))
            return print("Done with vis heatmap(Pred) on Training")

        def PCKh(info, pred_2D, gt_2D):
            pck_csv = pd.DataFrame()
            
            print(pred_2D)
            print(gt_2D)
            exit()

            # threshold = np.linspace(0, 1, num=20)
            # pck_dict = {}
            # pck_dict['threshold'] = threshold
            # pck_dict['average'] = np.zeros_like(threshold)

            return pck_csv 

    #Training
    # if conf.train_3D:


    #     learning_loss_val = loss_val_hg.clone().detach()
    #     learning_loss_val = torch.mean(learning_loss_val, dim=[1])

    #     loss_val_hg = torch.mean(loss_val_hg)
    #     epoch_val_hg.append(loss_val_hg.cpu().data.numpy())
        # vis_try(input, meta)
        # vis_try_ht(target)
        # exit()
        # 畫出heatmap
        # print(info[0][0])
        # print('index: ', meta['index'])
        # b =torch.zeros((64, 64))
        # for i in range(15):
        #     b+=info[0][i]

        # plt.imshow(b, cmap='magma')
        # plt.savefig("mygraph2.png")
        # exit()

        #畫出keypoints
        # print(i)
        # print(input.shape)
        # print(target.shape)
        # print(meta.keys()) # dict_keys(['index', 'center', 'scale', 'gt_3d', 'pts_crop'])
        # print(input[0])
        # print(meta['gt_3d'][0])
        
        # # for i in meta['gt_3d'][0]: 
        # #     print(i[:2])
        # import matplotlib.pyplot as plt
        # print(meta['index'][0])
        # for i in meta['gt_3d'][0]:
        #     plt.scatter(i[0], i[1])

        # plt.savefig("mygraph.png")

        #exit()

        #input_var = input.cuda(device=opt.device, non_blocking=True)
        #target_var = target.cuda(device=opt.device, non_blocking=True)


    # Training
    # for i, batch in enumerate(val_loader):
    #     #data_time.update(time.time() - end)
    #     '''
    #     input: torch.Size([32, 3, 256, 256])
    #     target: torch.Size([32, 16, 64, 64])

    #     '''
    #     input, target, meta= batch['input'], batch['target'], batch['meta']
    #     vis_joint_name(target[0])
    #     vis_try(input, meta)
    #     vis_try_ht(target)
    #     exit()
        # 畫出heatmap
        # print(info[0][0])
        # print('index: ', meta['index'])
        # b =torch.zeros((64, 64))
        # for i in range(15):
        #     b+=info[0][i]

        # plt.imshow(b, cmap='magma')
        # plt.savefig("mygraph2.png")
        # exit()

        #畫出keypoints
        # print(i)
        # print(input.shape)
        # print(target.shape)
        # print(meta.keys()) # dict_keys(['index', 'center', 'scale', 'gt_3d', 'pts_crop'])
        # print(input[0])
        # print(meta['gt_3d'][0])
        
        # # for i in meta['gt_3d'][0]: 
        # #     print(i[:2])
        # import matplotlib.pyplot as plt
        # print(meta['index'][0])
        # for i in meta['gt_3d'][0]:
        #     plt.scatter(i[0], i[1])

        # plt.savefig("mygraph.png")

        #exit()

        #input_var = input.cuda(device=opt.device, non_blocking=True)
        #target_var = target.cuda(device=opt.device, non_blocking=True)
        # 
if __name__ == '__main__':
  opt = opts().parse()
  conf = config() # get config_3D.py

  main(opt, conf)