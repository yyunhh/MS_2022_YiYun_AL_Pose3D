import logging

import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
####################################################
# 2D and 3D: concate 1D vector and 1D vector 
# 3D sigmoid
class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class LearnLoss_3D(nn.Module):
    def __init__(self,
                 linear_size=2048, #1024
                 num_stage=2,
                 p_dropout=0.5):
        super(LearnLoss_3D, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size =  2048 # 64
        # 3d joints
        self.output_size = 1 # 17 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size) #34, 1024
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout)) # 1024, 1024
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size) # 1024, 51

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)
        self.sigmoid = nn.Sigmoid() # 0428

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.sigmoid(y)
        return y


#####################################################    
class LearnLossActive(nn.Module):
    def __init__(self, num_feat, hg_feat, hg_depth, original=False, hg_feat_shape=(64, 32, 16, 8, 4)):
        '''
        :param num_feat:
        :param hg_feat:
        '''
        super(LearnLossActive, self).__init__()

        # List that houses the network
        self.fc_layers = []
        self.original = original
        self.hg_feat_shape = hg_feat_shape

        out = num_feat[-1]                              # out = 1
        num_feat = num_feat[:-1]

        if original:
            assert num_feat == [128, 64, 32, 16, 8]
            in_feat = hg_feat         # We are dealing with 256 x 5, not
        else:
            assert num_feat == [128, 64, 32, 16, 8]
            in_feat = hg_feat

        if not original:
            self.fc_layers.append(ConvolutionHourglassFeatureExtractor(4 + 1, hg_feat))

        for out_feat in num_feat:
            self.fc_layers.append(nn.Linear(in_features=in_feat, out_features=out_feat))
            self.fc_layers.append(nn.ReLU())
            in_feat = out_feat
        self.fc_layers.append(nn.Linear(in_features=in_feat, out_features=out))
        self.fc_layers = nn.ModuleList(self.fc_layers)


    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        encodings=None
        # GAP feature extractor
        if self.original:
            for layer in self.fc_layers:
                x = layer(x)
                if x.shape[-1] != 1:
                    encodings = x.clone().detach()
            return x, encodings

        # Conv feature extractor
        else:
            # Restoring heatmaps
            with torch.no_grad():
                conv_x = []
                border = 0

                for size in self.hg_feat_shape:
                    conv_x.append(x[:, :, border: border + (size**2)].reshape(x.shape[0], x.shape[1], size, size))
                    border += (size**2)

            x = self.fc_layers[0](conv_x)

            # [1:] skips the ConvFeatExtract layer
            for layer in self.fc_layers[1:]:
                x = layer(x)
                if x.shape[-1] != 1 and isinstance(layer, torch.nn.ReLU):
                    encodings = x.clone().detach()

            #return x, encodings
            return x


class ConvolutionHourglassFeatureExtractor(nn.Module):
    def __init__(self, depth, hg_feat):
        super(ConvolutionHourglassFeatureExtractor, self).__init__()
        self.hg_conv_feat_extract = []
        self.depth = depth

        # Down from 64 to 4
        for i in range(1, depth):    # 32 --> 16, 16 --> 8, 8 --> 4
            self.hg_conv_feat_extract.append(torch.nn.Conv2d(in_channels=hg_feat, out_channels=hg_feat,
                                                             kernel_size=(2, 2), stride=2, padding=0))
            self.hg_conv_feat_extract.append(torch.nn.ReLU())
        # Down from 4 to 1
        self.hg_conv_feat_extract.append(torch.nn.Conv2d(in_channels=hg_feat,
                                                         out_channels=hg_feat,
                                                         kernel_size=(4, 4), stride=1, padding=0))
        self.hg_conv_feat_extract.append(torch.nn.ReLU())

        self.hg_conv_feat_extract = nn.ModuleList(self.hg_conv_feat_extract)

    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        
        x_ = x[0]
        
        for i in range(self.depth - 1):
            
            x_ = self.hg_conv_feat_extract[2 * i](x_)
            x_ = self.hg_conv_feat_extract[(2 * i) + 1](x_)
            x_ = x[i+1] + x_
        
        out = self.hg_conv_feat_extract[-2](x_).squeeze()
        out = self.hg_conv_feat_extract[-1](out)
        
        return out

############ LearnLoss_mix ###############

# (1)改寫2D learnloss def__init__ 加入 linear 架構(ex: self.w)
# (2)改寫main_ver2.py 裡面 def load_models
# (3)改寫activelearning.py 裡面 def learning_loss_sample 架構
class LearnLoss_mix(nn.Module):
    def __init__(self, num_feat, hg_feat, hg_depth, 
                original=False, hg_feat_shape=(64, 32, 16, 8, 4), linear_size = 2304, num_stage = 2, p_dropout=0.5):
        '''
        :param num_feat:
        :param hg_feat:
        '''
        super(LearnLoss_mix, self).__init__()

        # List that houses the network
        self.fc_layers = []
        self.original = original
        self.hg_feat_shape = hg_feat_shape

        # linear arch: linear_size 以後的__init__都是
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        self.input_size =  2304 # 64
        self.output_size = 1 # 17 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size) #34, 1024
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout)) # 1024, 1024
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size) # 1024, 51

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)
        self.sigmoid = nn.Sigmoid() # 0428

        # linear arch

        out = num_feat[-1]                              # out = 1
        num_feat = num_feat[:-1]

        if original:
            assert num_feat == [128, 64, 32, 16, 8]
            in_feat = hg_feat         # We are dealing with 256 x 5, not
        else:
            assert num_feat == [128, 64, 32, 16, 8]
            in_feat = hg_feat

        if not original:
            self.fc_layers.append(ConvolutionHourglassFeatureExtractor(4 + 1, hg_feat).cuda())


    def forward(self, x, x2):
        '''

        :param x:
        :return:
        '''
        encodings=None
        # GAP feature extractor
        if self.original:
            for layer in self.fc_layers:
                x = layer(x)
                if x.shape[-1] != 1:
                    encodings = x.clone().detach()
            return x, encodings

        # Conv feature extractor
        else:
            # Restoring heatmaps
            with torch.no_grad():
                conv_x = []
                border = 0

                for size in self.hg_feat_shape:
                    conv_x.append(x[:, :, border: border + (size**2)].reshape(x.shape[0], x.shape[1], size, size))
                    border += (size**2)

            
            x = self.fc_layers[0](conv_x)
            
            y = torch.cat((x, x2), dim=1)

            y = self.w1(y)
            y = self.batch_norm1(y)
            y = self.relu(y)
            y = self.dropout(y)
            y = self.w2(y)
            y = self.sigmoid(y)
            return y
        
            # # [1:] skips the ConvFeatExtract layer
            # for layer in self.fc_layers[1:]:
            #     x = layer(x)
            #     if x.shape[-1] != 1 and isinstance(layer, torch.nn.ReLU):
            #         encodings = x.clone().detach()

            # return x, encodings


####################################################