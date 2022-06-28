'''
train obj for main_train_2D.py
把 train檔案放在這邊
'''
import cv2
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from utils import heatmap_loss
from utils_my import *

## vis
import matplotlib.pyplot as plt

def vis_try_ht_2(info, index):
    # heatmap # info放target
    b =torch.zeros((64, 64))
    for i in range(15): 
        b+=info[0][i]
    plt.imshow(b, cmap='magma')
    plt.savefig("../vis/Training_Pred_heatmap_{}.png".format(index))
    return print("Done with vis heatmap on Training")

# target and joints
def vis_try_ht_gt(info, index):
    # heatmap # info放target
    b =torch.zeros((64, 64))

    """
    h36m gt
    0: 'rfoot'
    1: 'rknee'
    2: 'rhip'
    3: 'lhip'
    4: 'lknee'
    5: 'lfoot'
    6: tor(兩個hip中間)
    7: spine
    8: neck
    9: head
    10: 'rhand'
    11: rwrist'
    12: rsho
    13: lsho
    14: lwrist
    15: lhand
    """

    """ 
    mpi3d gt



    """
    for i in range(15): 
        b+=info[i]
        plt.imshow(b, cmap='magma')
        plt.savefig('../vis/image_joint_{}_{}.png'.format(index, i))
    #cv2.imwrite('../vis/image_joint_{}.png'.format(index), b)
    # plt.imshow(b, cmap='magma')
    # plt.savefig("../vis/Training_Pred_heatmap_{}.png".format(index))
    return print("Done with vis heatmap on Training")

def vis_merge(img, info, index):
    # heatmap # info放target
    img = img.cpu().detach().numpy()
    img = (img[0]* 255).astype(np.uint8)
    plt.imshow(img)

    b =torch.zeros((64, 64))
    for i in range(15): 
        b+=info[0][i]
    plt.imshow(b, cmap='magma')
    plt.savefig("../vis/Training_Pred_heatmap_merge{}.png".format(index))
    return print("Done with vis heatmap on Training")

# merge的話需要用cv2 not plt
def vis_merge_cv(img, info, index):
    img = img.cpu().detach().numpy()
    img = (img[0]* 255).astype(np.uint8)
    cv2.imwrite('../vis/image_img_{}.png'.format(index), img)
    
    b =torch.zeros((64, 64))
    for i in range(15): 
        b+=info[0][i]

        # top_pic = cv2.resize(b, (256, 256), interpolation = cv2.INTER_AREA)
    print("check b shape", b.shape)
    b = (b.numpy()*255).astype(np.uint8)
    b = cv2.applyColorMap(b, cv2.COLORMAP_HOT)
    b = cv2.resize(b, (256, 256), interpolation = cv2.INTER_AREA)
    cv2.imwrite('../vis/image_heatmap_{}.png'.format(index), b)
    
    # overlapping_pic = cv2.addWeighted(b, 0.6, img, 0.4, 0) #(256, 256, 3) and unit8
    # cv2.imwrite('../vis/image_merge_{}.png'.format(index), overlapping_pic)

    return print("merge(CV)!")

# i: 確認joints order 跟 range 64*64 之間
def vis_target(info, index):
    print(">>> Start with target vis")

    b =torch.zeros((64, 64))
    for i in range(16): 
        b += info[0][i]
        b = b.cpu().detach().numpy()
        b = (b*255).astype(np.uint8)
        b = cv2.applyColorMap(b, cv2.COLORMAP_HOT)
        b = cv2.resize(b, (256, 256), interpolation = cv2.INTER_AREA)
        # plt.imshow(b, cmap='magma')
        cv2.imwrite('../vis/image_joint_{}_{}.png'.format(index, i), b)
    
    return print(">>> target vis finish")

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
        self.conf = conf
        

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

        idx_2D_to_3D_s = [16, 3, 4, 5, 2, 1, 0, 7, 6, 8, 9, 12, 11, 10, 13, 14, 15]

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
            print('Training for epoch: {}'.format(e+1))
            for (inp, out, meta, image) in tqdm(self.torch_dataloader):
                
                '''
                meta(dict): dict_keys(['index', 'center', 'scale', 'gt_3d', 'pts_crop', 'normalizer'])
                meta['gt_3d']: [B, 17, 3]
                out(heatmap): [B, 16, 64, 64]
                outputs(heatmap): [B, 16, 64, 64]
                '''

                # 0627: Delete later
                # print(">>>input vis start")
                # from datetime import datetime
                # dt = datetime.now()

                # input2 = inp.permute(0, 2, 3, 1)
                # input2 = input2.cpu().detach().numpy()
                # print("input shape: ", input2.shape) #input shape:  torch.Size([32, 256, 256, 3])
                # input2 = (input2[0]* 255).astype(np.uint8)
                # plt.imshow(input2)
                # plt.savefig("../vis/{}.jpg".format(dt))

                # print(">>>input vis finish")
                
                # 0627: Delete above later
               
                input2 = inp.permute(0, 2, 3, 1)
                input2 = input2.cuda(non_blocking=True)

                if self.conf.mpi_inf_3dhp:
                    target = out['heatmap_2D'].cuda()
                    # target = meta[:, :, 2].cuda().float()
                    # target = meta[0, :].cuda().float()
                    
                    # #target = out['original_ske'][0, :].cuda().float() # print("mpi3d target", target.shape) [17, 4]
                    # print("3D skeleton", target)
                    
                    # hm_gauss = 2 # kernel filter 參數
                    # heatmap_2D = np.zeros((16, 64, 64), dtype= np.float32) #(16, 64, 64)
                    # #heatmap_2D = np.zeros((16, 400, 400), dtype= np.float32) #(16, 400, 400): 才能塞的下orginal skeleton的range(-200~200)
                    
                    # # 每個joint 畫出對應的heatmap
                    # for i in range(16):
                    #     # 平移? 讓他們介於0~400之間
                    #     target[i][0]+=1
                    #     target[i][1]+=1
                    #     print(target[i][0])
                    #     print(target[i][1])
                    #     target[i] = target[i]*32
                    #     print("*32")
                    #     print(target[i][0])
                    #     print(target[i][1])
                    #     print("--")
                    #     heatmap_2D[i] = draw_gaussian(heatmap_2D[i], target[i], hm_gauss)

                    # target = heatmap_2D 
                    # print("heatmap shape", target.shape, type(target)) #(16, 64, 64)
                    
                else:
                    target = out.cuda(non_blocking=True)
                
                outputs, hourglass_features = self.network(input2) # [B, 16, 64, 64]
                #hourglass_features: <dict>, (['out', 1, 'feature_1', 2, 'feature_2', 3, 'feature_3', 4, 'feature_4', 5, 'feature_5', 'penultimate'])
                
                # 0627 Delete later

                # dt = datetime.now()
                # plt.clf()
                # vis_try_ht_2(outputs[0].detach().cpu().numpy(), dt)
                # # vis_merge(input2, outputs[0].detach().cpu().numpy(), dt)
                # # vis_merge_cv(input2, outputs[0].detach().cpu().numpy(), dt)

                # # print("outputs: ", outputs.shape)
                # # for i in range(16):
                # #     pred_2D = outputs[0, 0, :, :, :]
                # #     pred_2D = pred_2D.cpu().detach().numpy()
                # #     print("pred_2d: ", pred_2D.shape)
                # #     plt.scatter(pred_2D[i][0], pred_2D[i][1], c ="blue")
                # # plt.savefig("../vis/{}.jpg".format(dt))
                # # exit()
                # # 0627 Delete later


                # # 0627 detele later
                # # 將target存成一張張: 看index跟對應joint
                # # 對應joint放在vis_target大註解那裏
                # print("check target: ", target[0].shape) #H36M: torch.Size([16, 64, 64]) #MPI3D: [17]


                # dt = datetime.now()
                # plt.clf()

                # if self.conf.mpi_inf_3dhp:
                #     vis_try_ht_gt(target, dt)
                # else:
                #     vis_try_ht_gt(target[0].cpu().detach().numpy(), dt)
                # #vis_target(target[0], dt)

                
                # exit()
                # 0627 delete above later

                #outputs_go = outputs[:,1,:].detach().cpu().numpy() # type float32
                # print('outputs_go:', outputs_go.dtype)
                
                loss = heatmap_loss(outputs, target, self.n_stack)
                                
                
                # o_c = outputs.cpu().detach().numpy()

                # for k in range(12):
                #     self.vis_try_ht_2(o_c[k], meta['index'][k])
                
                
                # Will clear the gradients of hourglass
                self.optimizer.zero_grad()
            
                


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
        print("Model training completed\nBest validation loss (HG): {}\tBest Epoch: {}"
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
            print('Validation for epoch: {}'.format(e+1))

            for (inp_val, out_val, meta_val, image) in tqdm(self.val_dataloader):
            #for i, batch in enumerate(tqdm(self.val_dataloader)):
                # input_val, target_val, meta_val = batch['input'], batch['target'], batch['meta'] 


                

                input_val = inp_val.cuda(non_blocking=True)

                input_val = input_val.permute(0, 2, 3, 1) 

                outputs_val, hourglass_features_val = self.network(input_val)

                target_val = out_val['heatmap_2D'].cuda()

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
        
        print('2D-Pred:')
        print(emperical_loss.shape, emperical_loss)
        print('2D-GT:')
        print(true_loss.shape, true_loss)
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
                print('Error', true_loss.shape) #[4]
                print('Error2: ', gt_per_img.shape) #[4, 256, 3]
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
