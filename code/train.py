import os
import copy
import logging
import json

import numpy as np
import pandas as pd
from scipy import rand
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLROnPlateau

from config_3D import ParseConfig
from utils import visualize_image
from utils import heatmap_loss
from activelearning import ActiveLearning
from load_h36m import load_h36m
#from dataloader import load_hp_dataset
#from dataloader import Dataset_MPII_LSPET_LSP
#from evaluation import PercentageCorrectKeypoint
from models.learning_loss.LearningLoss import LearnLossActive, LearnLoss_3D, LearnLoss_mix, ConvolutionHourglassFeatureExtractor
from models.stacked_hourglass.StackedHourglass import PoseNet as Hourglass
#from main import Train

# 3D favor
from models.stacked_hourglass.Simple import Encoder, Decoder,  weight_init
from models.stacked_hourglass.c_vae import cVAE
import datetime
# 0406 simple 3D
from models.regression.yet_simple import LinearModel, weight_init
import models.regression.log as log
import models.regression.utils_simple as utils_simple
from torch.autograd import Variable
from progress.bar import Bar as Bar

import sys
#D:\Research\AL_Research\utils
# sys.path.insert(0, 'D:/Research/AL_Research/utils')
#from utils_3D import *
# from utils.utils import *
#from utilss.visualization import *
# from utils.camera import *
from dataloader_3D import * # #3D from Calvin
#from dataloader_my import * # my from hg-XX-3d

import torch.utils.data as data
import numpy as np
import time
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
        logging.warning('Could not load configuration! Exiting.')
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
      self.opt.data_dir, 'h36m', 'ECCV18_Challenge', split_, 'IMG') # change
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
    def __init__(self, h36m_dict, h36m_dict_2, h36m_dict_3, activelearning_obj, getitem_dump, conf, **kwargs): #0224_AL_Dataset
        self.conf = conf
        self.hm_shape = kwargs['hourglass']['hm_shape']
        self.hm_peak = kwargs['misc']['hm_peak']
        self.threshold = kwargs['misc']['threshold'] * self.hm_peak
        self.model_save_path = getitem_dump

        self.h36m = h36m_dict
        self.h36m_2 = h36m_dict_2
        self.h36m_3 = h36m_dict_3

        self.ind_to_jnt = list(ind_to_jnt.values())

        self.train_flag = False
        self.model_input_dataset = None

        activelearning_samplers = {
        'random': activelearning_obj.random,
        'coreset': activelearning_obj.coreset_sampling,
        'learning_loss': activelearning_obj.learning_loss_sampling,
        'entropy': activelearning_obj.multipeak_entropy,
        'mixture': activelearning_obj.mixture}

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

        for k in self.h36m.keys(): #0224_AL_Dataset
            for i in self.h36m_3[k]:
                self.h36m[k].append(i)
        #merge
        self.train_entire = self.h36m
        print('check', len(self.train_entire['input']))

        # add 1229 to
        del self.h36m_2
        del self.h36m_3 #0224_AL_Dataset

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
        
        if self.conf.active_learning_params['num_images']['total']==35822:
            print("Training with 100% dataset")
            self.train = self.train_entire
            print('The dataset number to train: ', len(self.train['pick_index']))
            logging.info('\nFinal size of Training Data: {}'.format(len(self.train['input']))) #self.train['index'].shape[0]
            self.input_dataset(train=True)
        
        else:

            print("Pick the indices")
            self.indices, self.pick_indices = activelearning_samplers[conf.active_learning_params['algorithm']](
                train=self.train_entire, dataset_size=self.dataset_size, mode=conf.active_learning_mode) # AL.py return 兩個參數

            # self.indices = activelearning_samplers[conf.active_learning_params['algorithm']](
            #     train=self.train_entire, dataset_size=self.dataset_size)
            # print(len(self.indices)) #4000
            # print(len(self.pick_indices)) #2000
            
            print("Turn into new dataset")
            print("0516 check", len(self.indices))
            print("0516 check train['input']", len(self.train_entire['pick_index']))
            ###
            # for index in self.indices:
            #     if index < 15000:
            #         merge_1 = merge_dataset_tmp() 
                
            #     if index > 15000 and index < 30000:
            #         merge_2 = merge_dataset_tmp() 

            #     else:
            #         merge_3 = merge_dataset_tmp()
            ###
            #0227
            not_indices_input = np.array(list(set(self.train_entire['pick_index'])-set(self.indices))) #pick_indices
            print("not_indices_input", not_indices_input.shape)
            
            self.train = {k : [val for i, val in enumerate(v) if i not in not_indices_input] for (k, v) in self.train_entire.items()}
            #self.train = self.merge_dataset(datasets=[self.train_entire], indices=[self.indices]) 
            #0227 註解換上行
            print('the # of training:', len(self.train['pick_index'])) 
            

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


    
    # def merge_dataset_tmp(self, datasets=None, indices=None):
    #     tmp_dataset ={}
    #     for key in datasets[0].keys():
    #         # print('indices', indices)
    #         # print('datasets: ', datasets)
    #         print("Do it", key)
    #         tmp_dataset[key] = np.concatenate([np.array(data[key])[index_] for index_, data in zip(indices, datasets)], axis=0)

    #     tmp_dataset['pick_index'] = np.arange(len(tmp_dataset'input']))#.shape[0])
    #     print("Merge complete!", tmp_dataset.keys())
    #     return tmp_dataset

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
        # learnloss_ = LearnLossActive(conf.learning_loss_fc, conf.args['hourglass']['inp_dim'], 4,
        #                              conf.learning_loss_original)

        if conf.learning_loss_obj=='2D':
            learnloss_ = LearnLossActive(conf.learning_loss_fc, conf.args['hourglass']['inp_dim'], 4,
                                     conf.learning_loss_original)
            path_ = ''  # best Learning Loss model
            learnloss_.load_state_dict(torch.load(
                conf.model_load_path_3D
                + '/checkpoint/best_ll_model_mode_{}'.format(conf.learning_loss_obj) #best_model_learnloss_{}
                + path_
                + '.pth', map_location='cpu'))
 
        if conf.learning_loss_obj=='3D':
            
            learnloss_ = LearnLoss_3D()
            path_ = ''
            learnloss_.load_state_dict(torch.load(
                    conf.model_load_path_3D
                    + '/checkpoint/best_ll_model_mode_{}'.format(conf.learning_loss_obj) #best_model_learnloss_{}
                    + path_
                    + '.pth', map_location='cpu'))

        if conf.learning_loss_obj=='2D+3D':

            learnloss_ = LearnLoss_mix(conf.learning_loss_fc, conf.args['hourglass']['inp_dim'], 4,
                                     conf.learning_loss_original)

            path_ = ''
            learnloss_.load_state_dict(torch.load(
                    conf.model_load_path_3D
                    + '/checkpoint/best_ll_model_mode_{}'.format(conf.learning_loss_obj) #best_model_learnloss_{}
                    + path_
                    + '.pth', map_location='cpu'))
                    
    else:
        logging.info('Defining the Learning Loss module. Training from scratch!')

        if conf.learning_loss_obj=='2D':
            learnloss_ = LearnLossActive(conf.learning_loss_fc, conf.args['hourglass']['inp_dim'], 4,
                                        conf.learning_loss_original)
        if conf.learning_loss_obj=='3D':
            # learnloss_ = LearnLoss_Linear()
            learnloss_ = LearnLoss_3D()

        if conf.learning_loss_obj=='2D+3D':
            learnloss_ = LearnLoss_mix(conf.learning_loss_fc, conf.args['hourglass']['inp_dim'], 4,
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
            #optim_dict = torch.load(conf.model_load_path + 'model_checkpoints/optim_best_model.tar') ##0224_M
            pass 

        else:
            assert type(conf.model_load_epoch) == int, "Load epoch for optimizer not specified"
            optim_dict = torch.load(conf.model_load_path + 'model_checkpoints/optim_epoch_{}.tar'.format(
                conf.model_load_epoch))

        # If the previous experiment used learn_loss, ensure the llal model is loaded, with the correct optimizer
        #assert optim_dict['learn_loss'] == conf.model_load_learnloss, "Learning Loss model needed to resume training" ##0224_M

        #hyperparameters['optimizer'].load_state_dict(optim_dict['optimizer_load_state_dict']) #0224_M
        logging.info('Optimizer state loaded successfully.\n')
        logging.info('Optimizer and Training parameters:\n')
    hyperparameters['loss_fn'] = torch.nn.MSELoss(reduction='none')

    return hyperparameters

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
        self.tb_writer = tb_writer    
        self.conf = conf
        self.opt = opt

        self.tb_writer = tb_writer
        self.train_learning_loss = conf.train_learning_loss
        self.model_save_path = conf.model_save_path

        self.total_steps = conf.total_steps
        self.epochs_3D = conf.epochs_3D
        self.batch_size_3D = conf.batch_size_3D
        self.idx_2D_to_3D = [16, 3, 4, 5, 2, 1, 0, 7, 6, 8, 9, 12, 11, 10, 13, 14, 15]

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

    def train_model_3D(self):
        print('Start training 3D model')
        print('Total steps - {}\t Batch Size - {}'.format(self.total_steps, self.batch_size_3D))
        best_val_network_3D = np.inf
        best_val_learnloss_3D = np.inf 
        best_epoch_network_3D = -1 
        best_epoch_learnloss_3D = -1 
        global_step =0

        loss_across_epochs = []
        validation_across_epochs = []

        total_loss_list, loss_func_list, kl_loss_list = [], [], [] # Calvin
        train_record = {"total_loss":[], "loss_func":[], "kl_loss": []} # Calvin

        for e in range(self.start_epoch, self.epochs_3D):

            epoch_loss = []
            epoch_loss_learnloss = []
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
                target_3D = meta['gt_3d'].cuda(device=self.opt.device, non_blocking=True) #x
                pred_2D, hourglass_features = self.network_2D(input_2D) #c


                '''
                input_2D:  <class 'torch.Tensor'> torch.Size([B, 256, 256, 3])
                target_3D:  <class 'torch.Tensor'> torch.Size([B, 17, 3])
                pred_2D: <class 'torch.Tensor'> torch.Size([B, 2, 16, 64, 64])
                pred_2D_max: torch.Size([B, 16, 2]) torch.float32
                target_3D: torch.Size([12, 16, 3]) torch.float32
                pred_3D: torch.Size([12, 48])

                pred_2D_train: torch.Size([32, 17, 2]
                target_3D: torch.Size([32, 17, 3]
                pred_3D: torch.Size([32, 17, 3]
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
                pred_2D_train = pred_2D_train/64 #0422
                pred_3D, mu, log_var = self.network_3D(target_3D, pred_2D_train) 
                pred_3D = torch.reshape(pred_3D, (self.conf.batch_size_3D, 17, 3)) #remember load dataset那邊也要改

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
                #print(total_loss.shape) #0403 test

                #LearnLoss
                #learning_loss_ = total_loss.clone().detach()
                #learning_loss_ = torch.mean(learning_loss_, dim=[1])
                if self.train_learning_loss:
                    # print("check learning loss input")
                    # print("input_2D", input_2D.shape)
                    # print('learning_loss(as?) ', learning_loss_)
                    # print('hourglass_features(as input) ', hourglass_features.keys())
                    #0416
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

            print()
            epoch_loss = sum(total_loss_list)/len(total_loss_list)
            self.tb_writer.add_scalar('Train', torch.Tensor([epoch_loss]), global_step)
    
            print("validation_3D starts:")
            validation_loss_hg = self.validation_3D(e)
            validation_learning_loss = 0 
            self.tb_writer.add_scalar('Validation/HG_Loss(3D)', torch.Tensor([validation_loss_hg]), global_step)

            # 0226 plot together
            self.tb_writer.add_scalars('Train stage', {'train':torch.Tensor([epoch_loss]), 'val':torch.Tensor([validation_loss_hg])}, global_step)

            # 0203要開
            # if self.train_learning_loss: 
            #     self.tb_writer.add_scalar('Validation/Learning_Loss', torch.Tensor([validation_learning_loss]), global_step)
            # 0203 end

            #Save the model

            # LearnLoss
            if self.train_learning_loss:
                epoch_loss_learnloss = np.mean(epoch_loss_learnloss)
                
            ##LearnLoss
            
            #torch.save(self.network_3D.state_dict(), self.model_save_path.format("model_epoch_{}.pth".format(e+1)))

            #if best_val_hg >epoch_loss: #total_loss # 0203
            if best_val_network_3D > validation_loss_hg: #total_loss
                torch.save(self.network_3D.state_dict(), self.model_save_path.format("best_3D_model.pth"))
                
                # torch.save(self.learnloss_network.state_dict(),
                #            self.model_save_path.format("best_model_learnloss_hg.pth"))

                #best_val_hg = epoch_loss #total_loss #0203
                best_val_network_3D = validation_loss_hg #total_loss
                best_epoch_network_3D = e + 1

                # torch.save({'epoch': best_epoch_hg,
                #             'optimizer_load_state_dict': self.optimizer.state_dict(),
                #             'mean_loss_train': epoch_loss,
                #             'mean_loss_validation': {'HG': validation_loss_hg, 'LearningLoss': validation_learning_loss},
                #             'learn_loss': self.train_learning_loss},
                #             self.model_save_path.format("optim_best_model.tar"))

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
                  "(Val: total loss) {}\t"
                  "(Best Model) {}".format(
                e+1,
                self.epochs_3D,
                epoch_loss,
                validation_loss_hg,
                best_epoch_network_3D))
            
            loss_across_epochs.append(epoch_loss)
            validation_across_epochs.append(validation_loss_hg)

            f = open(self.model_save_path.format("loss_data.txt"), "w")
            f_1 = open(self.model_save_path.format("validation_data.txt"), "w")
            f.write("\n".join([str(lsx) for lsx in loss_across_epochs]))
            f_1.write("\n".join([str(lsx) for lsx in validation_across_epochs]))
            f.close()
            f_1.close()

            if self.train_learning_loss:
                if best_val_learnloss_3D > epoch_loss_learnloss:
                    torch.save(self.learnloss_network.state_dict(), self.model_save_path.format("best_ll_model.pth"))
                # torch.save(self.learnloss_network.state_dict(),
                #            self.model_save_path.format("best_model_learnloss_hg.pth"))

                    best_val_learnloss_3D = epoch_loss_learnloss #total_loss
                    best_epoch_learnloss_3D = e + 1
                print("Loss at epoch {}/{}: \t"
                  "(Train: learnloss loss) {}\t"
                  "(Best Model) {}".format(
                e+1,
                self.epochs_3D,
                epoch_loss_learnloss,
                best_epoch_learnloss_3D))

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
                # 0215
                print("check learning loss input", hg_encodings['feature_5'].shape)
                exit()
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
        """
        e: epoch
        return: mean validation loss per batch for 3D model and learning loss.

        """
        # global_step_val =0 # 0203: what is this
        with torch.no_grad():
            # store the loss for all batches
            epoch_val_hg = [] #0203 這邊是指3D

            if self.train_learning_loss:
                epoch_val_learnloss = []
    
            self.network_2D.eval()
            self.network_3D.eval()
            
            if self.train_learning_loss:
                self.learnloss_network.eval()
            
            total_loss_val_list, loss_func_val_list, kl_loss_val_list = [], [], [] # Calvin
            train_record_val = {"total_loss":[], "loss_func":[], "kl_loss": []} # Calvin

            print('Validation for epoch: {}'.format(e+1))

            for (inp, out, meta, images) in tqdm(self.val_dataset_obj):
                
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
                pred_2D_train = pred_2D_train/64 # 0422
                pred_3D, mu, log_var = self.network_3D(target_3D, pred_2D_train)        
                #b_size = target_3D[0] #batch_size
                pred_3D = torch.reshape(pred_3D, (self.conf.batch_size_3D, 17, 3)) #b_size = 8 0203 

                criterion = nn.MSELoss(reduction='none')
                loss_func = criterion(pred_3D, target_3D)
                learning_loss_ = torch.mean(loss_func, (1,2)).clone().detach()
                loss_func = loss_func.mean()
  
                KL_loss = self.conf.kl_factor*-0.5*torch.sum(1+log_var - mu.pow(2) - log_var.exp())
                total_loss = loss_func + KL_loss
                total_loss = torch.mean(total_loss)
                epoch_val_hg.append(total_loss.cpu().data.numpy())
            print("Validation Loss HG at epoch {}/{}: {}".format(e+1, self.epochs_3D, np.mean(epoch_val_hg)))
            return np.mean(epoch_val_hg)

class Test_3D(object):
    '''
    Inference 3D model performance
    '''
    def __init__(self, network, network_3D, dataset_obj, conf, opt):
        self.network = network # load 2D model
        self.network_3D = network_3D
        self.dataset_obj = dataset_obj
        self.conf = conf
        self.opt = opt

        self.ind_to_jnt = {0: 'rfoot', 1: 'rknee', 2: 'rhip', 3: 'lhip', 4: 'lknee', 5: 'lfoot', 6: 'torso', 7: 'chest',
            8: 'Neck', 9: 'Head', 10: 'rhand', 11: 'relb', 12: 'rsho', 13:'lsho', 14: 'lelb', 15:'lhand'}
        self.model_save_path = conf.model_save_path
        self.idx_2D_to_3D = [16, 3, 4, 5, 2, 1, 0, 7, 6, 8, 9, 12, 11, 10, 13, 14, 15]

        self.device = opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
        # in main?
        # self.encoder = Encoder()
        # self.decoder = Decoder()
        # self.network_3D = cVAE(self.encoder, self.decoder).to(self.device)
        
        # print('Load 3D model from:', conf.model_load_path_3D)
        # self.network_3D.load_state_dict(torch.load(os.path.join(conf.model_load_path_3D, 'model_checkpoints/best_3D_model.pth')))
        # print('Done with loading the 3D model')

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

                if conf.test_3D_simple:
                    #pred_3D = self.network_3D.forward(torch.tensor(target_3D, device=opt.device), torch.tensor(outputs_val_pred, device=opt.device))
                    outputs_val_pred = torch.tensor(outputs_val_pred).cuda()
                    outputs_val_pred = outputs_val_pred.view(outputs_val_pred.size(0), -1)
                    #pred_2D_train = pred_2D_train.view(pred_2D_train.size(0), -1)
                    pred_3D = self.network_3D(outputs_val_pred)
                    pred_3D = pred_3D.reshape(pred_3D.size(0), 17, 3)

                else:        
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




def get_preds_s(hm, return_conf=False):
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

# train learning_loss 
def learning_loss_ver(hg_encodings, regression_encodings, learnloss_network, true_loss_3D, margin, gt_per_img, epoch,  obj, conf):
#def learning_loss_ver(hg_encodings, regression_encodings, learnloss_network, true_loss_2D, margin, gt_per_img, epoch,  obj):
    """
    hg_encodings: from 2D model
    regression_encodings(dict): from 3D model
    regression_encodings.keys(): [0, 1]
    regression_encodings[0].shape: torch.Size([32, 1024])

    """

    with torch.no_grad():
        if obj == '2D':
            encodings = torch.cat([hg_encodings['feature_5'].reshape(conf.batch_size_3D, hg_encodings['feature_5'].shape[1], -1),               # 64 x 64
                                    hg_encodings['feature_4'].reshape(conf.batch_size_3D, hg_encodings['feature_4'].shape[1], -1),               # 32 x 32
                                    hg_encodings['feature_3'].reshape(conf.batch_size_3D, hg_encodings['feature_3'].shape[1], -1),               # 16 x 16
                                    hg_encodings['feature_2'].reshape(conf.batch_size_3D, hg_encodings['feature_2'].shape[1], -1),               # 8 x 8
                                    hg_encodings['feature_1'].reshape(conf.batch_size_3D, hg_encodings['feature_1'].shape[1], -1)], dim=2)       # 4 x 4


        if obj == '3D':
            encodings = torch.cat((regression_encodings[0], regression_encodings[1]), dim=1) # torch.Size([32, 2048])



        
        if obj == '2D+3D':
            encodings_2D = torch.cat([hg_encodings['feature_5'].reshape(conf.batch_size_3D, hg_encodings['feature_5'].shape[1], -1),               # 64 x 64
                                    hg_encodings['feature_4'].reshape(conf.batch_size_3D, hg_encodings['feature_4'].shape[1], -1),               # 32 x 32
                                    hg_encodings['feature_3'].reshape(conf.batch_size_3D, hg_encodings['feature_3'].shape[1], -1),               # 16 x 16
                                    hg_encodings['feature_2'].reshape(conf.batch_size_3D, hg_encodings['feature_2'].shape[1], -1),               # 8 x 8
                                    hg_encodings['feature_1'].reshape(conf.batch_size_3D, hg_encodings['feature_1'].shape[1], -1)], dim=2)       # 4 x 4
            encodings_3D = torch.cat((regression_encodings[0], regression_encodings[1]), dim=1) # torch.Size([32, 2048])

            # print("2D shape: ", encodings_2D.shape) #  torch.Size([32, 256, 5456])
            # print("3D shape: ", encodings_3D.shape) #  torch.Size([32, 2048])


    #emperical_loss, encodings = learnloss_network(encodings)
    # if obj =='2D' or '3D': 
    #     emperical_loss = learnloss_network(encodings)
    #     emperical_loss = emperical_loss.squeeze()
    # can not put inside with_no_grad() 
    if obj =='2D+3D':
        emperical_loss = learnloss_network(encodings_2D, encodings_3D)
        emperical_loss = emperical_loss.squeeze()
    else:
        emperical_loss = learnloss_network(encodings)
        emperical_loss = emperical_loss.squeeze()

    # print("emperical_loss: ", emperical_loss.shape)
    # print("true_loss_3D: ", true_loss_3D.shape)
    assert emperical_loss.shape == true_loss_3D.shape, "Mismatch in Batch size for true and emperical loss"

    with torch.no_grad():
        gt_per_img = torch.sum(gt_per_img, dim=1)
        gt_per_img = torch.mean(gt_per_img, (1,2))
        gt_per_img += 0.1
        true_loss_3D = true_loss_3D / gt_per_img
        
        #split into pairs
        half_split = true_loss_3D.shape[0] // 2
        true_loss_i = true_loss_3D[: half_split]
        true_loss_j = true_loss_3D[half_split: 2 * half_split]

    emp_loss_i = emperical_loss[: (emperical_loss.shape[0] // 2)]
    emp_loss_j = emperical_loss[(emperical_loss.shape[0] // 2): 2 * (emperical_loss.shape[0] // 2)]    

    with torch.no_grad():
        true_loss_ = torch.cat([true_loss_i.reshape(-1, 1), true_loss_j.reshape(-1, 1)], dim=1)
        true_loss_scaled = true_loss_ / torch.sum(true_loss_, dim=1, keepdim=True)
        

    emp_loss_ = torch.cat([emp_loss_i.reshape(-1, 1), emp_loss_j.reshape(-1, 1)], dim=1)
    emp_loss_logsftmx = torch.nn.LogSoftmax(dim=1)(emp_loss_)
    llal_loss = torch.nn.KLDivLoss(reduction='batchmean')(input=emp_loss_logsftmx, target=true_loss_scaled)

    return torch.mean(llal_loss)



def train_model_3D_simple(train_loader, network_2D, network_3D, network_ll, criterion, optimizer, 
                        lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None, max_norm=True, kl_factor=None, conf=None):

    idx_2D_to_3D_s = [16, 3, 4, 5, 2, 1, 0, 7, 6, 8, 9, 12, 11, 10, 13, 14, 15]
    losses_simple = utils_simple.AverageMeter()
    network_3D.train()
    
    if conf.train_learning_loss:
        network_ll.train()

    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(train_loader)) #382 #0416

    print(lr_now)
    path_log = os.path.join(conf.model_save_path[:-20], 'checkpoint/prediction.txt')
    mpjpe_record_train = []
    learnloss_record_train=[]
    for (inp, out, meta, images) in tqdm(train_loader):    
        glob_step += 1

        # modify: lr_now通過以下之後變成none
        # if glob_step % lr_decay == 0 or glob_step == 1:
        #     lr_now = utils_simple.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)
        input_2D = inp.permute(0, 2, 3, 1)
        input_2D = input_2D.cuda(non_blocking=True)
        target_3D = meta['gt_3d'].cuda() #x 
        pred_2D, hourglass_features = network_2D(input_2D) #c
        np.save(file=conf.model_save_path.format('0428_encoding_2D.npy'), arr=hourglass_features) #0428

        pred_2D = pred_2D.cpu().detach().numpy().mean(axis= 1)
        pred_2D_max = get_preds_s(pred_2D) #(B, 16, 2)
        pred_2D_max = pred_2D_max.astype(np.float32)
        pred_2D_train = []
        for i in range(pred_2D_max.shape[0]):
            b_torse = (pred_2D_max[i][2]+ pred_2D_max[i][3])/2
            pred_2D_train.append(np.concatenate((pred_2D_max[i], [b_torse]), axis=0)[idx_2D_to_3D_s])
        pred_2D_train = np.array(pred_2D_train)
        pred_2D_train = torch.tensor(pred_2D_train).to(opt.device)

        # inputs = Variable(inps.cuda())
        # targets = Variable(tars.cuda(async=True))
        # outputs = model(inputs)
        # inputs -> pred_2D_train
        # targets -> target_3D
        # model -> network_3D
        # losses -> losses_simple
        pred_2D_train = Variable(pred_2D_train)
        target_3D = Variable(target_3D.cuda())
        

        # 0420 normalize
        pred_2D_train = pred_2D_train/64

        # 0421 start
        # gt_2D = target_3D[:, :, :2] #[32, 17, 2]
        # gt_2D = gt_2D.contiguous().view(gt_2D.size(0), -1)
        # outputs = network_3D(gt_2D)
        #0421 end
        if conf.train_3D:
            outputs, mu, log_var = network_3D(target_3D, pred_2D_train) 
        else:
            #print('pred_2D_train(original)', pred_2D_train.shape)# [32, 17, 2]
            pred_2D_train = pred_2D_train.view(pred_2D_train.size(0), -1)
            outputs, encoding_3D = network_3D(pred_2D_train) # 0421 open later since train with gt_2D # yet_simple.py: y_dict
            #np.save(file=conf.model_save_path.format('0428_encoding_3D.npy'), arr=encoding_3D) #0428

            # encode_vals = encoding_3D.values()
            # print(encoding_3D.keys())
            # print(type(encode_vals))
            # for a in encode_vals:
            #     print(a.shape) #(32, 1024)
            # exit()

            # print('pred_2D_train', pred_2D_train.shape)  #[32, 34]
            # print('pred_3D outputs', outputs.shape) # [32, 51]
            # print('pred_3D encoding', encoding_3D.keys()) #[0, 1]


        outputs = outputs.reshape(outputs.size(0), 17, 3)
    
        # mpjpe
        mpjpe_rec_train = mpjpe(outputs, target_3D).item()
        mpjpe_record_train.append(mpjpe_rec_train)
    


        #print("outputs shape: ", outputs.shape) #[32, 51] -> [32, 17, 3]
        #print("pred_3D shape: ", target_3D.shape) #[32, 17, 3]

        with open(path_log, 'w') as f:
            f.write(str(meta['index'][0]))
            f.write(str(pred_2D_train[0]))
            f.write(str(outputs[0]))
            f.write(str(target_3D[0]))

        # calculate loss
        
        if conf.train_3D:
            criterion_2 = nn.MSELoss(reduction='none')
            loss = criterion_2(outputs, target_3D)
            learning_loss_ = torch.mean(loss, (1,2)).clone().detach()
            KL_loss = kl_factor*-0.5*torch.sum(1+log_var - mu.pow(2) - log_var.exp())
            loss = loss.mean()
            total_loss = loss + KL_loss
            losses_simple.update(total_loss.item(), pred_2D_train.size(0))
            
            optimizer.zero_grad()
            total_loss.backward()

            if conf.train_learning_loss:
                print("train_learnloss")
                true_loss_3D = total_loss.clone().detach()
                #learning_loss_ver(hg_encodings, regression_encodings, learnloss_network, true_loss_3D, margin, gt_per_img, epoch,  obj):
                learnloss_loss = learning_loss_ver(hourglass_features, encoding_3D, network_ll, true_loss_3D, 0, 
                                                    input_2D, glob_step, conf.learning_loss_obj, conf)
                learnloss_loss.backward()
                learnloss_record_train.append(learnloss_loss.cpu().data.numpy())

                # pred_loss = network_ll(encoding_3D, )
                # loss_learnloss = learning_loss_ver(hourglass_features, learn_loss_, )
                # loss_learnloss.backward()

            # optimizer.zero_grad()
            # total_loss.backward()
            optimizer.step()
        
        else:
            optimizer.zero_grad()

            ##
            criterion_2 = nn.MSELoss(reduction='none')
            loss = criterion_2(outputs, target_3D)
            true_loss_3D = torch.mean(loss, (1,2)).clone().detach()
            loss = loss.mean()
            ## 


            #loss = criterion(outputs, target_3D)

            losses_simple.update(loss.item(), pred_2D_train.size(0))
            
            loss.backward()

            if conf.train_learning_loss:

                
                learnloss_loss = learning_loss_ver(hourglass_features, encoding_3D, network_ll, true_loss_3D, 0, 
                                                    input_2D, glob_step, conf.learning_loss_obj, conf)
                learnloss_loss.backward()
                learnloss_record_train.append(learnloss_loss.cpu().data.numpy())

            if max_norm:
                nn.utils.clip_grad_norm(network_3D.parameters(), max_norm=1)
            optimizer.step()

        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.4f}' \
            .format(batch=i + 1,
                    size=len(train_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses_simple.avg)
        bar.next()

    mpjpe_record_train = np.array(mpjpe_record_train)
    mpjpe_mean_train = mpjpe_record_train.mean()
    mpjpe_std_train = mpjpe_record_train.std()
    print(mpjpe_mean_train)
    print(mpjpe_std_train)
    
    learnloss_record_train = np.array(learnloss_record_train)
    learnloss_mean_train = learnloss_record_train.mean()
    print(learnloss_mean_train)

    bar.finish()
    if conf.train_learning_loss:
        return glob_step, lr_now, losses_simple.avg, mpjpe_mean_train, mpjpe_std_train, learnloss_mean_train
    else:
        return glob_step, lr_now, losses_simple.avg, mpjpe_mean_train, mpjpe_std_train, 0.0
    
def test_model_3D_simple(test_loader, network_2D, network_ll, network_3D, criterion, kl_factor, conf):
    idx_2D_to_3D_s = [16, 3, 4, 5, 2, 1, 0, 7, 6, 8, 9, 12, 11, 10, 13, 14, 15]
    with torch.no_grad():
        losses = utils_simple.AverageMeter()
        network_2D.eval()
        network_3D.eval()

        if conf.train_learning_loss:
            network_ll.eval()

        all_dist = []
        start = time.time()
        batch_time = 0

        mpjpe_record_test = []
        learnloss_record_test = []

        bar = Bar('>>>', fill='>', max=len(test_loader))
        
        for (inp, out, meta, images) in tqdm(test_loader):
            '''
            meta = {'index' : self.idxs[index], 'center' : c, 'scale' : s, 
                    'gt_3d': gt_3d, 'pts_crop': pts_crop, 'normalizer':normalizer}

            {'input': inp, 'target': out, 'meta': meta, 
                    'reg_target': reg_target, 'reg_ind': reg_ind, 'reg_mask': reg_mask, 'image':inp_copy}#, 'ht':out}
            '''
            input_2D = inp.permute(0, 2, 3, 1)
            input_2D = input_2D.cuda()
            target_3D = meta['gt_3d'].cuda() #x #meta['gt_3d']
            pred_2D, hourglass_features = network_2D(input_2D) #c

            pred_2D = pred_2D.cpu().detach().numpy().mean(axis= 1)
            pred_2D_max = get_preds_s(pred_2D) #(B, 16, 2)
            pred_2D_max = pred_2D_max.astype(np.float32)
            pred_2D_train = []

            for i in range(pred_2D_max.shape[0]):
                b_torse = (pred_2D_max[i][2]+ pred_2D_max[i][3])/2
                pred_2D_train.append(np.concatenate((pred_2D_max[i], [b_torse]), axis=0)[idx_2D_to_3D_s])
            pred_2D_train = np.array(pred_2D_train)
            pred_2D_train = torch.tensor(pred_2D_train).to(opt.device)
            # pred_2D_train = pred_2D_train.view(pred_2D_train.size(0), -1)
            
            pred_2D_train = Variable(pred_2D_train)
            target_3D = Variable(target_3D.cuda())
            pred_2D_train = pred_2D_train/64

            # 0421
            # gt_2D = target_3D[:, :, :2] #[32, 17, 2]
            # gt_2D = gt_2D.contiguous().view(gt_2D.size(0), -1)
            # outputs = network_3D(gt_2D)
            # 0421

            if conf.train_3D:
                outputs, mu, log_var = network_3D(target_3D, pred_2D_train) 
            else:

                pred_2D_train = pred_2D_train.view(pred_2D_train.size(0), -1)
                outputs, encoding_3D = network_3D(pred_2D_train) # 0421 open later since train with gt_2D

            outputs = outputs.reshape(outputs.size(0), 17, 3)
            mpjpe_rec_test = mpjpe(outputs, target_3D).item()
            mpjpe_record_test.append(mpjpe_rec_test)
            
            if conf.train_3D:
                criterion_2 = nn.MSELoss(reduction='none')
                loss = criterion_2(outputs, target_3D)
                learning_loss_ = torch.mean(loss, (1,2)).clone().detach()
                KL_loss = kl_factor*-0.5*torch.sum(1+log_var - mu.pow(2) - log_var.exp())
                loss = loss.mean()
                total_loss = loss + KL_loss
                losses.update(total_loss.item(), pred_2D_train.size(0))

            else:
                outputs_coord = outputs
                ###learnloss#
                criterion_2 = nn.MSELoss(reduction='none')
                true_loss_3D = criterion_2(outputs_coord, target_3D)
                true_loss_3D = torch.mean(true_loss_3D, (1, 2)).clone().detach()
                if conf.train_learning_loss:
                    learnloss_loss_test = learning_loss_ver(hourglass_features, encoding_3D, network_ll, true_loss_3D, 0, input_2D, 0, conf.learning_loss_obj, conf)
                    learnloss_record_test.append(learnloss_loss_test.cpu().data.numpy())

                ####
                loss = criterion(outputs_coord, target_3D)
                losses.update(loss.item(), pred_2D_train.size(0))

        mpjpe_record_test = np.array(mpjpe_record_test)

        mpjpe_mean_test = mpjpe_record_test.mean()
        mpjpe_std_test = mpjpe_record_test.std()
        print(mpjpe_mean_test)
        print(mpjpe_std_test)

        learnloss_record_test = np.array(learnloss_record_test)
        learnloss_mean_test = learnloss_record_test.mean()
        print(learnloss_mean_test)


        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.4f}' \
            .format(batch=i + 1,
                    size=len(test_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()

    bar.finish()
    if conf.train_learning_loss:
        return losses.avg, mpjpe_mean_test, mpjpe_std_test, learnloss_mean_test
    else:
        return losses.avg, mpjpe_mean_test, mpjpe_std_test, 0.0




def main(opt,conf):
    print('------Load info------')
    args = conf.args # get config_3D之後要打開->用在2D部分
    args_3D = conf.args_3D # args_3D['loss_type'] # MSE
    opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))

    # Load 2D model 
    hg_param = args['hourglass']
    print('Load the model from: ', conf.model_load_path)
    model, ll = load_models(conf=conf, load_hg=conf.model_load_hg, load_learnloss=conf.model_load_learnloss,
                                     best_model=conf.best_model, hg_param=hg_param, model_dir=conf.model_load_path)
   
    hyperparameters = define_hyperparams(conf, model, ll)
    writer = SummaryWriter(log_dir=os.path.join(conf.model_save_path[:-20], 'tensorboard'))

    # Load 3D model
    encoder = Encoder()
    decoder = Decoder()
    model_3D =  cVAE(encoder, decoder).to(opt.device)
    hyperparameter_3D = define_hyperparams_3D(conf, model_3D, ll)

    # Load simple 3D model(4/6)
    model_3D_simple = LinearModel()
    model_3D_simple.cuda()
    model_3D_simple.apply(weight_init)

    # AL 
    if conf.model_load_3D_model:
    #if conf.model_load_3D_model and (conf.active_learning_params['algorithm']=='learning_loss' or 'mixture'):
        print("Load previous 3D model to active learning")
        model_3D_simple_pre = LinearModel()
        model_3D_simple_pre.cuda()
        model_3D_simple_pre.apply(weight_init)
        ckpt_pre = torch.load(os.path.join(conf.model_load_path_3D, 'checkpoint/ckpt_best.pth.tar'))
        model_3D_simple_pre.load_state_dict(ckpt_pre['state_dict'])
        active_learning_obj = ActiveLearning(conf=conf,hg_network=model, learnloss_network=ll, network_3D =model_3D_simple_pre)
    else:
        active_learning_obj = ActiveLearning(conf=conf,hg_network=model, learnloss_network=ll, network_3D =model_3D_simple)

    # Load Data
    if conf.demo or conf.metric_3D or conf.pick_3D:
        pass
    else:
        h36m, h36m_2, h36m_3 = load_h36m(opt, 'train', precached_h36m = conf.precached_h36m)
        h36m_Data = H36M_AL(h36m_dict = h36m, h36m_dict_2 =h36m_2, h36m_dict_3 =h36m_3, 
                            activelearning_obj = active_learning_obj, getitem_dump =conf.model_save_path, conf = conf, **args)
        print("Done with active learning dataset")
  
        # create data indices for training and validation splits: 0203
        print("check", len(h36m_Data))
        split_totoal_indices = list(range(len(h36m_Data)))
        val_ratio = 0.2
        random_seed = 42
        
        split = int(np.floor(val_ratio*len(h36m_Data)))
        np.random.seed(random_seed)
        np.random.shuffle(split_totoal_indices)
        train_indices, val_indices = split_totoal_indices[split:], split_totoal_indices[:split]
        train_sampler = SubsetRandomSampler(train_indices) 
        valid_sampler = SubsetRandomSampler(val_indices)
        print(len(train_indices), len(valid_sampler))

        #0123 real batch size for 3D model hide here, need to modify the para as vars
        #0202 add pin_memory = True with ref: https://hackmd.io/@-CDCNK_qTUicXsissQsHMA/SJ6Gjpxv8#1-%E6%B8%9B%E5%B0%91-IO-%E6%99%82%E9%96%93
        #0224 conf.batch_size_3D
        torch_dataloader = torch.utils.data.DataLoader(h36m_Data , batch_size=32, pin_memory=True,
                                                                 num_workers=8, drop_last=True, sampler = train_sampler) #shuffle=True,
        val_loader = torch.utils.data.DataLoader(h36m_Data , batch_size=32, pin_memory=True,
                                                                 num_workers=8, drop_last=True, sampler = valid_sampler) #shuffle=True,


        del h36m
        del h36m_2
        del h36m_3
        print("Fininsh delelte, give memory back")

    # Train cVAE
    if conf.metric_3D: 
        val_loader = torch.utils.data.DataLoader(H36M(opt, 'val'), 
                                                batch_size=1, 
                                                shuffle=False,
                                                num_workers=8,
                                                pin_memory=True)


    criterion_3D_simple = nn.MSELoss(size_average=True).cuda()
    optimizer_3D_simple = torch.optim.Adam(model_3D_simple.parameters(), lr=1.0e-3) 

    logger_3D = log.Logger(os.path.join(conf.model_save_path[:-20], 'checkpoint/3D_simple_log.txt'))
    logger_3D.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'mpjpe_mean(train)', 'mpjpe_std(train)', 'mpjpe_mean(test)', 'mpjpe_std(test)', 'learnloss_train', 'learnloss_test'])
    print("Done with create log file")
        

    if conf.train_3D_simple or conf.train_3D:
        """
        所使用到的py檔案都放在regression資料夾裡面
        1) train_model_3D_simple: 完成torch_loader裡面的資料
        2) opt file: 建立model_3D_simple config 存所有變數
        3) load and resume: skip
        4)
        """
        glob_step = 0
        err_best = 10000
        start_epoch = 0
        end_epoch = 250

        lr_now = 1.0e-3
        lr_init = 1.0e-3
        lr_decay = 100000
        lr_gamma = 0.96

        ### learnloss
        err_best_ll = 100
        ### learnloss

        # model_3D_simple train:


        if conf.train_3D_simple:
            model_3D = model_3D_simple
        if conf.train_3D:
            model_3D =  model_3D
        lr_3D = 0.001
        kl_factor = 0.0001
        criterion_3D = nn.MSELoss(size_average=True).cuda()
        optimizer_3D = torch.optim.Adam(model_3D.parameters(), lr=lr_3D) 
        ckpt_path = os.path.join(conf.model_save_path[:-20], 'checkpoint/')

        for epoch in range(start_epoch, end_epoch):
            print('==========================')
            print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

            glob_step, lr_now, loss_train, mpjpe_mean_train, mpjpe_std_train, learnloss_train= train_model_3D_simple(
                torch_dataloader, network_2D = model, network_3D = model_3D, network_ll = ll, criterion = criterion_3D, optimizer = optimizer_3D,
                lr_init=lr_init, lr_now=lr_now, glob_step=glob_step, lr_decay=lr_decay, gamma=lr_gamma,
                max_norm=True, kl_factor =kl_factor, conf=conf)

            loss_test, mpjpe_mean_test, mpjpe_std_test, learnloss_test = test_model_3D_simple(val_loader, network_2D = model, network_3D = model_3D, network_ll = ll, criterion = criterion_3D, kl_factor = kl_factor, conf = conf)
        
            print(epoch + 1, lr_now, loss_train, loss_test, loss_train)
            # update log file
            logger_3D.append([epoch + 1, lr_now, loss_train, loss_test, mpjpe_mean_train, mpjpe_std_train, mpjpe_mean_test, mpjpe_std_test, learnloss_train, learnloss_test],
                        ['int', 'float', 'float', 'flaot', 'float', 'float', 'float', 'float', 'float', 'float'])

            # modify: save model 還沒寫
            """
            model -> model_3D_simple
            optimizer -> optimizer_3D_simple

            """
            ############### save learnloss model
            is_best_ll = learnloss_test < err_best_ll
            err_best_ll = min(learnloss_test, err_best_ll)
            if is_best_ll:
                torch.save(ll.state_dict(), ckpt_path + "best_ll_model_mode_{}.pth".format(conf.learning_loss_obj))
            
            ############### save learnloss model
            is_best = loss_test < err_best
            err_best = min(loss_test, err_best)
            
            if is_best:
                if conf.train_3D:
                    torch.save(model_3D.state_dict(), ckpt_path + "best_3D_model.pth") #0422 add last model
                else:
                    log.save_ckpt({'epoch': epoch + 1,
                                'lr': lr_now,
                                'step': glob_step,
                                'err': err_best,
                                'state_dict': model_3D_simple.state_dict(),
                                'optimizer': optimizer_3D_simple.state_dict()},
                                ckpt_path = os.path.join(conf.model_save_path[:-20], 'checkpoint/'), 
                                #ckpt_path='checkpoint/',
                                is_best=True)
                print("Best model:", epoch+1)

            else:
                if conf.train_3D:
                    torch.save(model_3D.state_dict(), ckpt_path + "last_3D_model.pth")
                else:
                    log.save_ckpt({'epoch': epoch + 1,
                                'lr': lr_now,
                                'step': glob_step,
                                'err': err_best,
                                'state_dict': model_3D_simple.state_dict(),
                                'optimizer': optimizer_3D_simple.state_dict()},
                                ckpt_path = os.path.join(conf.model_save_path[:-20], 'checkpoint/'), 
                                #ckpt_path='checkpoint/',
                                is_best=False)

        logger_3D.close()

    #0406 simple 3D model
    #0406 test simple 3D model
    if conf.test_3D_simple:
        test_loader = torch.utils.data.DataLoader(H36M(opt, 'val'), 
        batch_size=1, 
        shuffle=False,
        num_workers=8,
        pin_memory=True)
        print(">>>load the pre-trained 3D model simple:")
        ckpt = torch.load('checkpoint/ckpt_best.pth.tar')
        model_3D_simple.load_state_dict(ckpt['state_dict'])
        # optimizer_3D_simple.load_state_dict(ckpt['optimizer'])
        # err_set = []

        print(">>> Test model 3D simple:")
        metric_3D_obj = Test_3D(network = model, network_3D = model_3D_simple, dataset_obj=test_loader, conf = conf, opt= opt)
        eval_results_test = metric_3D_obj.inference()
        print("MPJPE: {}, MPJPE_STD: {}".format(eval_results_test['mpjpe'][0], eval_results_test['mpjpe'][1]))
        exit()
    # 0406 test simple 3D model
    
    # if conf.train_3D:
    #     global_step = 0 
    #     err_best = 0
    #     start_epoch = 0
    #     end_epoch = 500
    #     lr_now = 1.0e-3
    #     lr_init = 1.0e-3
    #     lr_decay = 100000
    #     lr_gamma = 0.96
        
    #     idx_2D_to_3D_s = [16, 3, 4, 5, 2, 1, 0, 7, 6, 8, 9, 12, 11, 10, 13, 14, 15]
        



    #     print('Start training 3D network-------')
    #     # self, network_2D, network_3D, learnloss_network, hyperparameters, dataset_obj, val_dataset_obj, tb_writer, conf, opt):
    #     train_3D_obj = Train_3D(network_2D = model, network_3D = model_3D, learnloss_network=ll, hyperparameters=hyperparameter_3D,
    #                         dataset_obj=torch_dataloader, conf=conf, tb_writer=writer, opt= opt, val_dataset_obj=val_loader)
    #     train_3D_obj.train_model_3D()
    #     print('Done with training 3D model')

    if conf.metric_3D:
        print('Start inference 3D network-------')
        
        print('Load 3D model from:', conf.model_load_path_3D)
        model_3D.load_state_dict(torch.load(os.path.join(conf.model_load_path_3D, 'model_checkpoints/best_3D_model.pth')))
        
        metric_3D_obj = Test_3D(network = model, network_3D = model_3D, dataset_obj=val_loader, conf = conf, opt= opt)
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
        exit()



    if conf.train:
        print('Start training----')
        train_obj = Train2(network=model, learnloss=ll, hyperparameters=hyperparameters,
                            dataset_obj=torch_dataloader, conf=conf, tb_writer=writer, opt= opt, val_dataloader=val_loader)
        train_obj.train_model()
        print('Done with training----')


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
            return pck_csv 
if __name__ == '__main__':
  opt = opts().parse()
  conf = config() # get config_3D.py

  main(opt, conf)