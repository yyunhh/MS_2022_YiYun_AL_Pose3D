
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
from load_h36m import load_h36m, H36M_AL, H36M
from models.learning_loss.LearningLoss import LearnLossActive
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
from datetime import datetime
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
import matplotlib.animation as animation
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
        
        today = datetime.now
        new_run = "../vis/{}" .format(today().strftime('%Y%m%d_%H-%M'))
        os.makedirs(new_run)
        img_save_path = os.path.join(new_run)
        
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

                print(target_val[0].shape)
                
                

                #add and switch
                outputs_val_pred = []
                for i in range(pred_val.shape[0]):
                    b_torse = (pred_val[i][2]+pred_val[i][3])/2
                    outputs_val_pred.append(np.concatenate((pred_val[i], [b_torse]), axis=0)[self.idx_2D_to_3D])
                c+=1


                if conf.test_3D_simple:
                    outputs_val_pred = torch.tensor(outputs_val_pred).cuda()
                    outputs_val_pred = outputs_val_pred.view(outputs_val_pred.size(0), -1)
                    outputs_val_pred = outputs_val_pred/64

                    pred_3D = self.network_3D(outputs_val_pred)
                    pred_3D = pred_3D.reshape(pred_3D.size(0), 17, 3)

                else:
                    outputs_val_pred_new = [x/64 for x in outputs_val_pred]
                    #pred_3D, mu, log_var = self.network_3D.forward(torch.tensor(target_3D, device=opt.device), torch.tensor(outputs_val_pred, device=opt.device))
                    pred_3D, mu, log_var = self.network_3D.forward(torch.tensor(target_3D, device=opt.device), torch.tensor(outputs_val_pred_new, device=opt.device))
                
                b_size = 1
                pred_3D = torch.reshape(pred_3D, (b_size, 17, 3))
                
                f_img_info = os.path.join(img_save_path, 'img_info.txt')
                if c == 12:
                #if c //12 ==0:
                    print('index')
                    print(meta_val['index'])
                    # print('2D-GT')
                    # print(target_val2, target_val2.shape)
                    # print('2D-pred')
                    # print(pred_val, pred_val.shape)

                    # print('Modify 2D-pred')
                    # pred_val = torch.tensor(pred_val, device=opt.device)
                    # print(pred_val, pred_val.shape)
                    with open(f_img_info, 'a+') as f:

                        f.write('Index:')
                        f.write(str(meta_val['index']))
                        f.write('\n3D-GT:\n')
                        f.write(str(target_3D))
                        f.write('\n3D-Pred:\n')
                        f.write(str(pred_3D))
                        f.write('\n-------------------')
                        f.close()
                        
                    print('3D-GT')
                    print(target_3D, target_3D.shape)
                    print('3D-Pred')
                    print(pred_3D)

                    save_3D(pred_3D.cpu().numpy(), meta_val['index'], img_save_path)
                    save_3D(target_3D.cpu().numpy(), meta_val['index'], img_save_path, gt= True)
                    save_3D_together(target_3D.cpu().numpy(), pred_3D.cpu().numpy(), meta_val['index'], img_save_path)
                    

                if c== 1440 :
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
                    save_3D(pred_3D.cpu().numpy(), meta_val['index'], img_save_path)
                    save_3D(target_3D.cpu().numpy(), meta_val['index'], img_save_path, gt= True)
                    save_3D_together(target_3D.cpu().numpy(), pred_3D.cpu().numpy(), meta_val['index'], img_save_path)

                mpjpe_rec = mpjpe(pred_3D, target_3D).item()

                if c < 200: 
                    print("mpjpe ", mpjpe_rec)
                    if mpjpe_rec > 200:
                        vis_try_ht_gt(target_val[0].cpu().detach().numpy(), meta_val['index'])
                        # vis_try_ht_2(outputs_val2[0].cpu().detach().numpy(), meta_val['index'])
                        save_3D(pred_3D.cpu().numpy(), meta_val['index'], img_save_path)
                        save_3D(target_3D.cpu().numpy(), meta_val['index'], img_save_path, gt= True)
                        save_3D_together(target_3D.cpu().numpy(), pred_3D.cpu().numpy(), meta_val['index'], img_save_path)


                # if c==1440:
                #     print(mpjpe_rec)

                # if mpjpe_rec < 80:
                #     save_3D(pred_3D.cpu().numpy(), meta_val['index'], img_save_path)
                #     save_3D(target_3D.cpu().numpy(), meta_val['index'], img_save_path, gt= True)
                #     save_3D_together(target_3D.cpu().numpy(), pred_3D.cpu().numpy(), meta_val['index'], img_save_path)

                mpjpe_record.append(mpjpe_rec)
            
            fill_size = len(str(max_batch))
            print("\rProgress: "+str(c+1).zfill(fill_size)+"/"+str(max_batch), end="")
            
        print("\nFinish")
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
    print('------Load info------')
    args = conf.args # get config_3D之後要打開->用在2D部分
    args_3D = conf.args_3D # args_3D['loss_type'] # MSE
    opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))

    # model place here
    hg_param = args['hourglass']

    # 2D model
    print('Load the model from: ', conf.model_load_path)
    model, ll = load_models(conf=conf, load_hg=conf.model_load_hg, load_learnloss=conf.model_load_learnloss,
                                     best_model=conf.best_model, hg_param=hg_param, model_dir=conf.model_load_path)
   
    hyperparameters = define_hyperparams(conf, model, ll)
    writer = SummaryWriter(log_dir=os.path.join(conf.model_save_path[:-20], 'tensorboard'))

    ## AL 
    active_learning_obj = ActiveLearning(conf=conf,hg_network=model, learnloss_network=ll)

    if conf.demo or conf.metric_3D or conf.pick_3D or conf.test_3D_simple or conf.metric_2D:
        pass
    else:
        h36m, h36m_2, h36m_3 = load_h36m(opt, 'train', precached_h36m = conf.precached_h36m)
        #print('main_h36m: ', h36m.keys()) #(['input', 'target', 'meta', 'reg_target', 'reg_ind', 'reg_mask', 'image'])
        h36m_Data = H36M_AL(h36m_dict = h36m, h36m_dict_2 =h36m_2, h36m_dict_3 =h36m_3, activelearning_obj = active_learning_obj, getitem_dump =conf.model_save_path, conf = conf, **args)
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
        #end

        #0123 real batch size for 3D model hide here, need to modify the para as vars
        #0202 add pin_memory = True with ref: https://hackmd.io/@-CDCNK_qTUicXsissQsHMA/SJ6Gjpxv8#1-%E6%B8%9B%E5%B0%91-IO-%E6%99%82%E9%96%93
        #0224 conf.batch_size_3D
        torch_dataloader = torch.utils.data.DataLoader(h36m_Data , batch_size=32, pin_memory=True,
                                                                 num_workers=8, drop_last=True, sampler = train_sampler) #shuffle=True,
        val_loader = torch.utils.data.DataLoader(h36m_Data , batch_size=32, pin_memory=True,
                                                                 num_workers=8, drop_last=True, sampler = valid_sampler) #shuffle=True,

    
        print("Give the memory back!")
        del h36m
        del h36m_2
        del h36m_3
        print("Fininsh delelte!")
        

    # load 3D model
    encoder = Encoder()
    decoder = Decoder()
    model_3D =  cVAE(encoder, decoder).to(opt.device)
    hyperparameter_3D = define_hyperparams_3D(conf, model_3D, ll)

    #0406 simple 3D model
    model_3D_simple = LinearModel()
    model_3D_simple.cuda()
    model_3D_simple.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model_3D_simple.parameters()) / 1000000.0))
    
    criterion_3D_simple = nn.MSELoss(size_average=True).cuda()
    optimizer_3D_simple = torch.optim.Adam(model_3D_simple.parameters(), lr=1.0e-3) #lr=opt.lr

    logger_3D_simple = log.Logger(os.path.join('checkpoint/', '3D_simple_log.txt'))
    logger_3D_simple.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'err_test'])

    print("Done with create log file")
        

    if conf.train_3D_simple:
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
        end_epoch = 300

        lr_now = 1.0e-3
        lr_init = 1.0e-3
        lr_decay = 100000
        lr_gamma = 0.96

        # model_3D_simple train:
        idx_2D_to_3D_s = [16, 3, 4, 5, 2, 1, 0, 7, 6, 8, 9, 12, 11, 10, 13, 14, 15]

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

        def train_model_3D_simple(train_loader, network_2D, network_3D, criterion, optimizer, 
                                lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None, max_norm=True):

            #opt.device

            losses_simple = utils_simple.AverageMeter()
            model_3D_simple.train()

            start = time.time()
            batch_time = 0
            bar = Bar('>>>', fill='>', max=len(train_loader)) #382

            print(lr_now)

            for (inp, out, meta, images) in tqdm(train_loader):    
                glob_step += 1

                # modify: lr_now通過以下之後變成none
                # if glob_step % lr_decay == 0 or glob_step == 1:
                #     lr_now = utils_simple.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)
                input_2D = inp.permute(0, 2, 3, 1)
                input_2D = input_2D.cuda(non_blocking=True)
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
                

                # inputs = Variable(inps.cuda())
                # targets = Variable(tars.cuda(async=True))
                # outputs = model(inputs)
                # inputs -> pred_2D_train
                # targets -> target_3D
                # model -> network_3D
                # losses -> losses_simple
                
                #print("check with 2D prediction shape", pred_2D_train.shape) #[32, 17, 2]
                #print("check pred_2D_train size(0)", pred_2D_train.size(0))

                pred_2D_train = Variable(pred_2D_train)
                target_3D = Variable(target_3D.cuda())

                #print(network_3D)
                pred_2D_train = pred_2D_train.view(pred_2D_train.size(0), -1)
                outputs = network_3D(pred_2D_train)
                # modify
                outputs = outputs.reshape(outputs.size(0), 17, 3)
                #print("outputs shape: ", outputs.shape) #[32, 51] -> [32, 17, 3]
                # modify
                
                #print("pred_3D shape: ", target_3D.shape) #[32, 17, 3]

                # calculate loss
                optimizer.zero_grad()
                loss = criterion(outputs, target_3D)
                losses_simple.update(loss.item(), pred_2D_train.size(0))
                loss.backward()
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

            bar.finish()
            print(lr_now)
            return glob_step, lr_now, losses_simple.avg
            
        def test_model_3D_simple(test_loader, network_2D, network_3D, criterion):
            with torch.no_grad():
                losses = utils_simple.AverageMeter()
                network_2D.eval()
                network_3D.eval()

                all_dist = []
                start = time.time()
                batch_time = 0

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

                    
                    pred_2D_train = Variable(pred_2D_train)
                    target_3D = Variable(target_3D.cuda())


                    pred_2D_train = pred_2D_train.view(pred_2D_train.size(0), -1)
                    outputs = network_3D(pred_2D_train)
                    outputs = outputs.reshape(outputs.size(0), 17, 3)

                    outputs_coord = outputs
                    loss = criterion(outputs_coord, target_3D)
                    losses.update(loss.item(), pred_2D_train.size(0))
                    
                    
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
            return losses.avg

        for epoch in range(start_epoch, end_epoch):
            print('==========================')
            print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

            # per epoch

            # glob_step, lr_now, loss_train = train_model_3D_simple(
            #     torch_dataloader, network_2D = model, network_3D = model_3D_simple, criterion=criterion_3D_simple, optimizer = optimizer_3D_simple)
            # exit()
            glob_step, lr_now, loss_train = train_model_3D_simple(
                torch_dataloader, network_2D = model, network_3D = model_3D_simple, criterion = criterion_3D_simple, optimizer = optimizer_3D_simple,
                lr_init=lr_init, lr_now=lr_now, glob_step=glob_step, lr_decay=lr_decay, gamma=lr_gamma,
                max_norm=True)

            loss_test = test_model_3D_simple(val_loader, network_2D = model, network_3D = model_3D_simple, criterion = criterion_3D_simple)
            
            print(epoch + 1, lr_now, loss_train, loss_test, loss_train)
            # update log file
            logger_3D_simple.append([epoch + 1, lr_now, loss_train, loss_test, loss_train],
                        ['int', 'float', 'float', 'flaot', 'float'])

            # modify: save model 還沒寫
            """
            model -> model_3D_simple
            optimizer -> optimizer_3D_simple

            """
            is_best = loss_test < err_best
            err_best = min(loss_test, err_best)
            if is_best:
                log.save_ckpt({'epoch': epoch + 1,
                            'lr': lr_now,
                            'step': glob_step,
                            'err': err_best,
                            'state_dict': model_3D_simple.state_dict(),
                            'optimizer': optimizer_3D_simple.state_dict()},
                            ckpt_path='checkpoint/',
                            is_best=True)
                print("Best model:", epoch+1)

            else:
                log.save_ckpt({'epoch': epoch + 1,
                            'lr': lr_now,
                            'step': glob_step,
                            'err': err_best,
                            'state_dict': model_3D_simple.state_dict(),
                            'optimizer': optimizer_3D_simple.state_dict()},
                            ckpt_path='checkpoint/',
                            is_best=False)

        logger_3D_simple.close()


    if conf.test_3D_simple:
        test_loader = torch.utils.data.DataLoader(H36M(opt, 'val'), 
        batch_size=1, 
        shuffle=False,
        num_workers=0,
        pin_memory=True)

        print(">>>load the pre-trained 3D model simple:")
        
        ckpt = torch.load(os.path.join(conf.model_load_path_3D, 'checkpoint/ckpt_last.pth.tar'))
        #ckpt = torch.load('checkpoint/ckpt_last.pth.tar')
        
        model_3D_simple.load_state_dict(ckpt['state_dict'])
        # optimizer_3D_simple.load_state_dict(ckpt['optimizer'])
        # err_set = []

        print(">>> Test model 3D simple:")
        metric_3D_obj = Test_3D(network = model, network_3D = model_3D_simple, dataset_obj=test_loader, conf = conf, opt= opt)
        eval_results_test = metric_3D_obj.inference()
        print("model from: ", conf.model_load_path_3D)
        print("MPJPE(last): {}, MPJPE_STD: {}".format(eval_results_test['mpjpe'][0], eval_results_test['mpjpe'][1]))

        # best
        ckpt = torch.load(os.path.join(conf.model_load_path_3D, 'checkpoint/ckpt_best.pth.tar'))
        model_3D_simple.load_state_dict(ckpt['state_dict'])
        metric_3D_obj = Test_3D(network = model, network_3D = model_3D_simple, dataset_obj=test_loader, conf = conf, opt= opt)
        eval_results_test = metric_3D_obj.inference()
        print("model from: ", conf.model_load_path_3D)
        print("MPJPE(best): {}, MPJPE_STD: {}".format(eval_results_test['mpjpe'][0], eval_results_test['mpjpe'][1]))


if __name__ == '__main__':
  opt = opts().parse()
  conf = config() # get config_3D.py
  main(opt, conf)