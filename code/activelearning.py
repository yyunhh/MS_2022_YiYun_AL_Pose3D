import os
import cv2
import torch
import torch.utils.data
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from skimage.feature import peak_local_max
from scipy.special import softmax as softmax_fn
from scipy.stats import entropy as entropy_fn

# ## 1002
# class Activelearning(object):
#     def __init__(self, conf, opt, hg_network, learnloss_network): #3D favor
#         self.hg_network = hg_network
#         self.conf = conf
#         self.opt = opt
#         self.learnloss_network = learnloss_network
#         #self.network_3D = network_3D


#     def random_sampler(self, dataset):
#         # load之前的index

#         if self.conf.resume_training:
#             return np.load(os.path.join(self.conf.model_load_path, 'model_checkpoints/annotation.npy'))

#         if self.conf.model_load_hg == True:
#             annotated_idx = np.load(os.path.join(self.conf.model_load_path, 'model_checkpoints/annotation.npy'))
#         else:
#             annotated_idx = np.array([])

#         """
#         random select the index
#         remain = total - index(before) 
#         random select conf.number 
#         """
#         #remain_index = np.array(dataset.index - annotated_idx) 
#         unlabel_index = np.array(list(set(dataset.index) - set(annotated_idx)))
#         sample_index = np.random.choice(unlabel_index, size= self.conf.active_learning_params['num_images']['total'])
#         annotated_idx = np.concatenate([annotated_idx, sample_index],axis = 0)
#         # np.save(annotated_idx, 'annotated_idx.npy')
#         np.save(file = self.conf.model_save_path.format('annotated_idx.npy'), arr = annotated_idx)
#         return annotated_idx

#     def learning_loss_sampler(self, dataset):
#         if self.conf.model_load_HG:
#             annotated_idx = np.load('annotated.npy')
#         else:
#             annotated_idx = np.array([])
            
#         unlabel_index = np.array(list(set(dataset.index) - set(annotated_idx)))

#         unlabel_dataset = {}

#         for key in dataset.keys():
#             unlabel_dataset[key] = dataset[key][unlabel_index] #activelearning_debug.ipynb
        
#         """
#         0) Unlabel_dataset->dataloader
#         1) load unlabel dataset to learning_loss model
#         2) sorting
#         3) restore to annotated dataset
#         """

#         # 0)
#         Dataset_ = ActiveLearning_Dataset(unlabel_dataset)
#         al_dataloader = torch.utils.data.Dataloader(Dataset_, batch_size=self.conf.batch_size, shuffle=False, num_worker = 0)

#         # 1)
#         #self.learnloss_network.eval() 在哪邊出場?

#         with torch.no_grad(): # 忘了加這項!
#             for images in tqdm(al_dataloader):

#                 images = images.to(non_blocking=True, device='cuda')
#                 _, hourglass_features = self.hg_network(images)

#                 if self.conf.learning_loss_original:
#                     encodings = hourglass_features['penultimate']
#                 else:
#                     encodings = torch.cat(
#                         [hourglass_features['feature_5'].reshape(images.shape[0], hourglass_features['feature_5'].shape[1], -1),
#                          hourglass_features['feature_4'].reshape(images.shape[0], hourglass_features['feature_4'].shape[1], -1),
#                          hourglass_features['feature_3'].reshape(images.shape[0], hourglass_features['feature_3'].shape[1], -1),
#                          hourglass_features['feature_2'].reshape(images.shape[0], hourglass_features['feature_2'].shape[1], -1),
#                          hourglass_features['feature_1'].reshape(images.shape[0], hourglass_features['feature_1'].shape[1], -1)], dim=2)

#                 learnloss_pred_, _ = self.learnloss_network(encodings)
#                 learnloss_pred_ = learnloss_pred_.squeeze()

                
#                 learnloss_pred = learnloss_pred_.cpu()


#                 # unlabel_score data structure? -> array or list or set
                
#         """
#         2) sorting
#         (score, index)
#         sort the score
#         """
#         pred_with_index = np.concatenate([learnloss_pred.numpy().reshape(-1, 1),
#                                           unlabel_index.reshape(-1, 1)], axis=-1)

#         pred_with_index = pred_with_index[pred_with_index[:, 0].argsort()]
#         indices = pred_with_index[-self.conf.active_learning_params['num_images']['total']:, 1]

#         annotated_idx = np.concatenate([annotated_idx, indices], axis=0).astype(np.int32)

#         unique, counts = np.unique(dataset['dataset'][annotated_idx], return_counts=True)
#         proportion = {key: value for (key, value) in zip(unique, counts)}

#         np.save(file=self.conf.model_save_path.format('annotation.npy'), arr=annotated_idx)

#         return annotated_idx

#         """
#         3) restore to annotated_index
#         """


        


# class ActiveLearning_Dataset(torch.utils.data.Dataset):
#     def __init__(self, dataset_dict, indices=None):
    
#         if indices is None:
#             self.images = dataset_dict['img']
#             self.bounding_box = dataset_dict['bbox_coords']
#         else:
#             self.images = dataset_dict['img'][indices]
#             self.bounding_box = dataset_dict['bbox_coords'][indices]

#         self.xy_to_uv = lambda xy: (xy[1], xy[0])

#     def __len__(self):
#         return self.images.shape[0]

#     def __getitem__(self, item):
#         '''

#         :param item:
#         :return:
#         '''

#         image =  self.images[item]
#         bounding_box = self.bounding_box[item]

#         # Determine crop
#         img_shape = np.array(image.shape)

#         # Bounding box for the first person
#         [min_x, min_y, max_x, max_y] = bounding_box[0]

#         tl_uv = self.xy_to_uv(np.array([min_x, min_y]))
#         br_uv = self.xy_to_uv(np.array([max_x, max_y]))
#         min_u = tl_uv[0]
#         min_v = tl_uv[1]
#         max_u = br_uv[0]
#         max_v = br_uv[1]

#         centre = np.array([(min_u + max_u) / 2, (min_v + max_v) / 2])
#         height = max_u - min_u
#         width = max_v - min_v

#         scale = 2.0

#         top_left = np.array([centre[0] - (scale * height / 2), centre[1] - (scale * width / 2)])
#         bottom_right = np.array([centre[0] + (scale * height / 2), centre[1] + (scale * width / 2)])

#         top_left = np.maximum(np.array([0, 0], dtype=np.int16), top_left.astype(np.int16))
#         bottom_right = np.minimum(img_shape.astype(np.int16)[:-1], bottom_right.astype(np.int16))

#         # Cropping the image
#         image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]

#         # Resize the image
#         image = self.resize_image(image, target_size=[256, 256, 3])

#         return torch.tensor(data=image / 256.0, dtype=torch.float32, device='cpu')

#     def resize_image(self, image_=None, target_size=None):
#         '''

#         :return:
#         '''
#         # Compute the aspect ratios
#         image_aspect_ratio = image_.shape[0] / image_.shape[1]
#         tgt_aspect_ratio = target_size[0] / target_size[1]

#         # Compare the original and target aspect ratio
#         if image_aspect_ratio > tgt_aspect_ratio:
#             # If target aspect ratio is smaller, scale the first dim
#             scale_factor = target_size[0] / image_.shape[0]
#         else:
#             # If target aspect ratio is bigger or equal, scale the second dim
#             scale_factor = target_size[1] / image_.shape[1]

#         # Compute the padding to fit the target size
#         pad_u = (target_size[0] - int(image_.shape[0] * scale_factor))
#         pad_v = (target_size[1] - int(image_.shape[1] * scale_factor))

#         output_img = np.zeros(target_size, dtype=image_.dtype)

#         # Write scaled size in reverse order because opencv resize
#         scaled_size = (int(image_.shape[1] * scale_factor), int(image_.shape[0] * scale_factor))

#         padding_u = int(pad_u / 2)
#         padding_v = int(pad_v / 2)

#         im_scaled = cv2.resize(image_, scaled_size)
#         # logging.debug('Scaled, pre-padding size: {}'.format(im_scaled.shape))

#         output_img[padding_u : im_scaled.shape[0] + padding_u,
#                    padding_v : im_scaled.shape[1] + padding_v, :] = im_scaled

#         return output_img

# EGL sampling
from utils import heatmap_loss
from utils import shannon_entropy
from utils import heatmap_generator

class ActiveLearning(object):
    '''
    Contains collection of active learning algorithms for human joint localization
    '''

    def __init__(self, conf, hg_network, network_3D, learnloss_network):
        self.conf = conf
        self.hg_network = hg_network
        self.network_3D = network_3D
        self.learnloss_network = learnloss_network

        self.hg_network.eval()
        self.network_3D.eval()
        self.learnloss_network.eval()

    def ll_core(self, train, dataset_size, hg_depth=4):
        '''
        Learning loss sampling of images from training dataset
        :param train: (dict) Training dataset
        :param dataset_size: (dict) Stores the size of each dataset - MPII / LSP+LSPET
        :return: (np.ndarray) Indices chosen for sampling
        '''
        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model_load_path, 'model_checkpoints/annotation.npy'))

        if self.conf.model_load_hg:
            annotated_idx = np.load(os.path.join(self.conf.model_load_path_3D, 'model_checkpoints/annotation.npy'))
        else:
            annotated_idx = np.array([])

        print("----------Learning Loss approach begin----------")

        print("Previous # of annotation is: ", len(annotated_idx))
        # Set of indices not annotated
        unlabelled_idx = np.array(list(set(train['pick_index'])-set(annotated_idx)))
        unlabelled_dataset = {}

        for unlabel_idx in range(10000):
            unlabelled_dataset['input']=np.array(train['input'])[unlabelled_idx]
        print("the # of unlabel dataset goto learnloss to select: ", len(unlabelled_dataset['input']))
        #1229
        # for key in train.keys():
        #     unlabelled_dataset[key] = np.array(train[key])[unlabelled_idx]

        print('unlabelled_dataset: ', type(unlabelled_dataset), unlabelled_dataset.keys()) #dict_keys(['input', 'target', 'meta', 'reg_target', 'reg_ind', 'reg_mask', 'image', 'pick_index'])
        print(len(unlabelled_dataset['input'])) #8
        

        #dataset_ = ActiveLearningDataset(dataset_dict = unlabelled_dataset) #1229
        dataset_ = ActiveLearningDataset(dataset_dict = train)
        learnloss_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.batch_size, shuffle=False, num_workers=0)

        learnloss_pred = None

        # Prediction and concatenation of the learning loss network outputs
        with torch.no_grad():
            for images in tqdm(learnloss_dataloader):

                images = images.to(non_blocking=True, device='cuda')

                _, hourglass_features = self.hg_network(images)

                if self.conf.learning_loss_original:
                    # encodings = torch.cat([hourglass_features[depth] for depth in range(1, hg_depth + 2)], dim=-1)
                    encodings = hourglass_features['penultimate']

                else:
                    # No longer concatenating, will now combine features through convolutional layers
                    encodings = torch.cat(
                        [hourglass_features['feature_5'].reshape(images.shape[0], hourglass_features['feature_5'].shape[1], -1),
                         hourglass_features['feature_4'].reshape(images.shape[0], hourglass_features['feature_4'].shape[1], -1),
                         hourglass_features['feature_3'].reshape(images.shape[0], hourglass_features['feature_3'].shape[1], -1),
                         hourglass_features['feature_2'].reshape(images.shape[0], hourglass_features['feature_2'].shape[1], -1),
                         hourglass_features['feature_1'].reshape(images.shape[0], hourglass_features['feature_1'].shape[1], -1)], dim=2)

                learnloss_pred_, _ = self.learnloss_network(encodings)
                learnloss_pred_ = learnloss_pred_.squeeze()

                try:
                    learnloss_pred = torch.cat([learnloss_pred, learnloss_pred_.cpu()], dim=0)
                except TypeError:
                    learnloss_pred = learnloss_pred_.cpu()

        # argsort defaults to ascending
        pred_with_index = np.concatenate([learnloss_pred.numpy().reshape(-1, 1),
                                          unlabelled_idx.reshape(-1, 1)], axis=-1)

        pred_with_index = pred_with_index[pred_with_index[:, 0].argsort()]
        indices = pred_with_index[-self.conf.active_learning_params['num_images']['total']:, 1]

        annotated_idx = np.concatenate([annotated_idx, indices], axis=0).astype(np.int32)

        # unique, counts = np.unique(train['dataset'][annotated_idx], return_counts=True)
        # proportion = {key: value for (key, value) in zip(unique, counts)}
        # with open(self.conf.model_save_path.format('sampling_proportion.txt'), "x") as file:
        #     file.write('Learning Loss sampling\n')
        #     [file.write("{}: {}\n".format(key, proportion[key])) for key in proportion.keys()]

        np.save(file=self.conf.model_save_path.format('annotation.npy'), arr=annotated_idx)

        return annotated_idx, indices


        
    def random(self, train, dataset_size, mode):
        '''
        Randomly samples images from training dataset
        :param train: (dict) Training dataset
        :param dataset_size: (dict) Stores the size of each dataset - MPII / LSP+LSPET
        :return: (np.ndarray) Indices chosen for sampling
        '''
        print("-------------------Random Method-------------------------")
        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model_load_path, 'model_checkpoints/annotation.npy'))

        # Load previously annotated images indices
        if self.conf.train:
            if self.conf.model_load_hg:
                annotated_idx = np.load(os.path.join(self.conf.model_load_path, 'model_checkpoints/annotation.npy'))
            else:
                annotated_idx = np.array([])

        if self.conf.train_3D or self.conf.metric_3D or self.conf.train_3D_simple:
            if self.conf.model_load_3D_model:
                annotated_idx = np.load(os.path.join(self.conf.model_load_path_3D, 'model_checkpoints/annotation.npy'))
                print("Previous number of images: ", len(annotated_idx))
            else:
                annotated_idx = np.array([])

        # Set of indices not annotated
        unlabelled_idx = np.array(list(set(train['pick_index'])-set(annotated_idx)))

        print("unlabelled_idx: ", unlabelled_idx.shape) #35822

        num_images = self.conf.active_learning_params['num_images'] # 總共要標的照片total
        # num_images(dict)
        print("num_images: ", num_images['total']) #35822


        # Determine if per dataset sampling or overall
        if self.conf.args['mpii_only']:
            overall_annotate = np.random.choice(unlabelled_idx, size=num_images['total'], replace= False) #0224: turn replace->True

            # Update annotated images indices
            annotated_idx = np.concatenate([annotated_idx, overall_annotate], axis=0).astype(np.int32)

        else:
            # Separation index between datasets
            accum_lspet = dataset_size['lspet']['train']
            accum_lsp = dataset_size['lspet']['train'] + dataset_size['lsp']['train']

            # Find indices which are not annotated for each dataset
            lspet_unlabelled = unlabelled_idx[np.where(np.logical_and(unlabelled_idx >= 0, unlabelled_idx < accum_lspet))]
            lsp_unlabelled = unlabelled_idx[np.where(np.logical_and(unlabelled_idx >= accum_lspet, unlabelled_idx < accum_lsp))]

            # Randomly sample images from each dataset
            lspet_annotated = np.random.choice(lspet_unlabelled, size=num_images['lspet'], replace=False)
            lsp_annotated = np.random.choice(lsp_unlabelled, size=num_images['lsp'], replace=False)

            # Update annotated images indices
            annotated_idx = np.concatenate([annotated_idx, lspet_annotated, lsp_annotated], axis=0).astype(np.int32)

        #0321 bug
        # unique, counts = np.unique(np.array(train['pick_index'])[annotated_idx], return_counts=True) #0224_???
        # #unique, counts = np.unique(train['dataset'][annotated_idx], return_counts=True)
        # proportion = { key: value for (key, value) in zip(unique, counts)}
        # with open(self.conf.model_save_path.format('sampling_proportion.txt'), "x") as file:
        #     file.write('Random sampling\n')
        #     [file.write("{}: {}\n".format(key, proportion[key])) for key in proportion.keys()]
        # print("save!")

        print("anntotated_idx: ",annotated_idx.shape)
        np.save(file=self.conf.model_save_path.format('annotation.npy'), arr=annotated_idx)
        
        return annotated_idx, overall_annotate

    def goto_learning_loss(self, train, coreset_index, hg_depth = 4):

        idx_2D_to_3D_s = [16, 3, 4, 5, 2, 1, 0, 7, 6, 8, 9, 12, 11, 10, 13, 14, 15]
        print("# of coreset select: ", len(coreset_index))
        np.save(file=self.conf.model_save_path.format('(core_ll)coreset_annotation.npy'), arr=coreset_index)

        unlabelled_idx = np.array(list(set(coreset_index)))
        unlabelled_dataset = {k : [val for i, val in enumerate(v) if i in unlabelled_idx] for (k, v) in train.items()}
        dataset_ = ActiveLearningDataset(dataset_dict = unlabelled_dataset)
        learnloss_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.batch_size, shuffle=False, num_workers=8)
        learnloss_pred = None
        #####0516
        # unlabelled_idx = coreset_index
        # unlabelled_dataset = train #0226
        # print("Done with merge new coreset dataset") #0123
        # dataset_ = ActiveLearningDataset(unlabelled_dataset)
        # learnloss_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.batch_size, shuffle=False, num_workers=8)
        # learnloss_pred = None
        # print("Turn into coreset dataloader and go to ll:") #0123
        # print("# of unlabelled data to select: ", len(unlabelled_dataset['input']))
        # unlabelled_idx = np.array(unlabelled_dataset['pick_index'])
        #####0516
        # Prediction and concatenation of the learning loss network outputs
        print("Let's go learnloss with ", self.conf.learning_loss_obj)
        with torch.no_grad():
            for images in tqdm(learnloss_dataloader):

                images = images.to(non_blocking=True, device='cuda')

                _, hourglass_features = self.hg_network(images)
                ####### 3D pred and encoding #######
                _ = _.cpu().detach().numpy().mean(axis=1)
                _max = self.get_preds_s(_)
                _max = _max.astype(np.float32)
                _max_train = []

                for i in range(_max.shape[0]):
                    b_torse = (_max[i][2]+ _max[i][3])/2
                    _max_train.append(np.concatenate((_max[i], [b_torse]), axis=0)[idx_2D_to_3D_s])

                _max_train = np.array(_max_train)
                _max_train = torch.tensor(_max_train).to('cuda')
                _max_train = _max_train/64
                _max_train = _max_train.view(_max_train.size(0), -1)
                _output, _encoding_3D = self.network_3D(_max_train)
                ##################################

                if self.conf.learning_loss_obj=='2D':
                    # # encodings = torch.cat([hourglass_features[depth] for depth in range(1, hg_depth + 2)], dim=-1)
                    # encodings = hourglass_features['penultimate']

                    # No longer concatenating, will now combine features through convolutional layers
                    encodings = torch.cat(
                        [hourglass_features['feature_5'].reshape(images.shape[0], hourglass_features['feature_5'].shape[1], -1),
                         hourglass_features['feature_4'].reshape(images.shape[0], hourglass_features['feature_4'].shape[1], -1),
                         hourglass_features['feature_3'].reshape(images.shape[0], hourglass_features['feature_3'].shape[1], -1),
                         hourglass_features['feature_2'].reshape(images.shape[0], hourglass_features['feature_2'].shape[1], -1),
                         hourglass_features['feature_1'].reshape(images.shape[0], hourglass_features['feature_1'].shape[1], -1)], dim=2)
                    learnloss_pred_ = self.learnloss_network(encodings)

                if self.conf.learning_loss_obj == '3D':
                    encodings = torch.cat((_encoding_3D[0], _encoding_3D[1]), dim=1)
                    learnloss_pred_ = self.learnloss_network(encodings)

                if self.conf.learning_loss_obj == '2D+3D':
                    
                    encodings_2D = torch.cat(
                        [hourglass_features['feature_5'].reshape(images.shape[0], hourglass_features['feature_5'].shape[1], -1),
                         hourglass_features['feature_4'].reshape(images.shape[0], hourglass_features['feature_4'].shape[1], -1),
                         hourglass_features['feature_3'].reshape(images.shape[0], hourglass_features['feature_3'].shape[1], -1),
                         hourglass_features['feature_2'].reshape(images.shape[0], hourglass_features['feature_2'].shape[1], -1),
                         hourglass_features['feature_1'].reshape(images.shape[0], hourglass_features['feature_1'].shape[1], -1)], dim=2)

                    encodings_3D = torch.cat((_encoding_3D[0], _encoding_3D[1]), dim=1)
                    learnloss_pred_ = self.learnloss_network(encodings_2D, encodings_3D)
                    # concate 再一起並且丟進learnloss_network裡面

                #learnloss_pred_, _ = self.learnloss_network(encodings)
                #learnloss_pred_ = self.learnloss_network(encodings)
                ############## 3D loss prediction #####################
                # _, learnloss_pred = self.learnloss_network(encodings)
                #######################################################
                learnloss_pred_ = learnloss_pred_.squeeze()

                try:
                    learnloss_pred = torch.cat([learnloss_pred, learnloss_pred_.cpu()], dim=0)
                except TypeError:
                    learnloss_pred = learnloss_pred_.cpu()

        # argsort defaults to ascending

        print("check 1", len(learnloss_pred.numpy().reshape(-1, 1))) # 3793
        print("check 2", len(unlabelled_idx.reshape(-1, 1))) # 4000
        
        pred_with_index = np.concatenate([learnloss_pred.numpy().reshape(-1, 1),
                                          unlabelled_idx.reshape(-1, 1)], axis=-1)

        pred_with_index = pred_with_index[pred_with_index[:, 0].argsort()]
        #indices = pred_with_index[-self.conf.active_learning_params['num_images']['total']:, 1] #0123
        indices = pred_with_index[-len(unlabelled_idx)//2:, 1] 
        print("Done with the next learning loss sample")
        del learnloss_dataloader
        np.save(file=self.conf.model_save_path.format('goto_ll_idx.npy'), arr=indices)
        return indices
###################################################################
    def goto_coreset(self, train, ll_idx, mode):
        print("--------------goto coreset------------------")
        if self.conf.model_load_3D_model: #.model_load_hg: #0224_M
            print("Load the annotation") #0123 delete later
            annotated_idx = np.load(os.path.join(self.conf.model_load_path_3D, 'model_checkpoints/annotation.npy'))
        else:
            annotated_idx = np.array([])

        print("# of previous select: ", len(annotated_idx))

        def update_distances(cluster_centers, encoding, min_distances=None):
            '''
            Based on: https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
            Update min distances given cluster centers.
            Args:
              cluster_centers: indices of cluster centers
              only_new: only calculate distance for newly selected points and update
                min_distances.
              rest_dist: whether to reset min_distances.
            '''

            if len(cluster_centers) != 0:
                # Update min_distances for all examples given new cluster center.
                x = encoding[cluster_centers]
                
                
                # x_ll = encoding[ll_idx] # 0518
                
                #print("Check the cluster center type: ", x.shape, type(x))
                np.save(file=self.conf.model_save_path.format('center_indices.npy'), arr=cluster_centers) #0125
                np.save(file=self.conf.model_save_path.format('center_encoding.npy'), arr=x) #0125
                
                dist = pairwise_distances(encoding, x, metric='euclidean')
                #dist = pairwise_distances(x_ll, x, metric='euclidean') #0518
                
                
                if min_distances is None:
                    min_distances = np.min(dist, axis=1).reshape(-1, 1)
                else:
                    min_distances = np.minimum(min_distances, dist)

            return min_distances

        print("# of learnloss select: ", len(ll_idx))
        np.save(file=self.conf.model_save_path.format('(ll_core)learnloss_annotation.npy'), arr=ll_idx)
        
        unlabelled_idx = np.array(list(set(ll_idx)))
        unlabelled_idx = unlabelled_idx.astype(int)

        #unlabelled_dataset = {k : [val for i, val in enumerate(v) if i in unlabelled_idx] for (k, v) in train.items()}
        #dataset_ = ActiveLearningDataset(dataset_dict = unlabelled_dataset)
        dataset_ = ActiveLearningDataset(dataset_dict = train)
        coreset_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.batch_size, shuffle=False, num_workers=8)
        hg_encoding = None

        # Part 1: Obtain embeddings
        # Disable autograd to speed up inference
        with torch.no_grad():
            for images in tqdm(coreset_dataloader):
                
                images = images.to(non_blocking=True, device='cuda')
                _, hourglass_features = self.hg_network(images)

                try:
                    hg_encoding = torch.cat((hg_encoding, hourglass_features['penultimate'].cpu()), dim=0)
                except TypeError:
                    hg_encoding = hourglass_features['penultimate'].cpu()

        hg_final_encoding = hg_encoding.squeeze().numpy()
        print("Finsh complete Core-Set encoding")
        print("Check hg_encoding type: ", hg_final_encoding.shape, type(hg_final_encoding))
        np.save(file=self.conf.model_save_path.format('hg_encoding.npy'), arr=hg_final_encoding) #0125
        
        logging.info('Core-Set encodings computed.')

        # Part 2: k-Centre Greedy
        core_set_budget = self.conf.active_learning_params['num_images']['total']
        min_distances = None

        if len(annotated_idx) != 0:
            min_distances = update_distances(cluster_centers=annotated_idx, encoding=hg_final_encoding, min_distances=None)
        coreset_idx =np.array([])
        for _ in tqdm(range(core_set_budget)):
            if len(annotated_idx) == 0:  # Initial choice of point
                # Initialize center with a randomly selected datapoint
                ind = np.random.choice(np.arange(hg_final_encoding.shape[0]))
            else:
                # ind = np.argmax(min_distances)
                
                # 0  1  2  3  4  5 
                # 00 11 22 33 44 55
                # 1 2 3
                # 
                # 最大值且為index
                # min_distance = min_distances[unlabelled_idx]
                # ind = np.argmax(min_distance)

                ind = 0
                max_dist = 0

                for i in unlabelled_idx:
                    if i == 35822:
                        unlabelled_idx = np.delete(unlabelled_idx, np.argwhere(unlabelled_idx == 35822))
                for i in unlabelled_idx:
                    tmp = min_distances[i]
                    if(tmp > max_dist):
                        max_dist = tmp
                        ind = i
                unlabelled_idx = np.delete(unlabelled_idx, np.argwhere(unlabelled_idx == ind))
                
                
                #print("ind check", ind.shape, type(ind), ind) #一個: <class 'numpy.int64'> 16193

            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            min_distances = update_distances(cluster_centers=[ind], encoding=hg_final_encoding, min_distances=min_distances)
            coreset_idx = np.concatenate([coreset_idx, [ind]], axis=0).astype(np.int32) #0123
        print("# of coreset select", len(coreset_idx))
        np.save(file=self.conf.model_save_path.format('(ll_core)coreset_annotation.npy'), arr=coreset_idx)
        return coreset_idx
########################################
    def coreset_sampling(self, train, dataset_size, mode):
        '''
        Coreset sampling of images from training dataset
        :param train: (dict) Training dataset
        :param dataset_size: (dict) Stores the size of each dataset - MPII / LSP+LSPET
        :return: (np.ndarray) Indices chosen for sampling

        '''
        print("-------------------Coreset Method--------------------------")
        def update_distances(cluster_centers, encoding, min_distances=None):
            '''
            Based on: https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
            Update min distances given cluster centers.
            Args:
              cluster_centers: indices of cluster centers
              only_new: only calculate distance for newly selected points and update
                min_distances.
              rest_dist: whether to reset min_distances.
            '''

            if len(cluster_centers) != 0:
                # Update min_distances for all examples given new cluster center.
                x = encoding[cluster_centers]
                np.save(file=self.conf.model_save_path.format('center_indices.npy'), arr=cluster_centers) #0125
                np.save(file=self.conf.model_save_path.format('center_encoding.npy'), arr=x) #0125
                dist = pairwise_distances(encoding, x, metric='euclidean')

                if min_distances is None:
                    min_distances = np.min(dist, axis=1).reshape(-1, 1)
                else:
                    min_distances = np.minimum(min_distances, dist)

            return min_distances

        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model_load_path_3D, 'model_checkpoints/annotation.npy')) 

        if self.conf.model_load_3D_model: #.model_load_hg: #0224_M
            print("Load the annotation") #0123 delete later
            annotated_idx = np.load(os.path.join(self.conf.model_load_path_3D, 'model_checkpoints/annotation.npy'))
        else:
            annotated_idx = np.array([])
        
        dataset_ = ActiveLearningDataset(train)
        coreset_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.batch_size, shuffle=False, num_workers=8) ## 0123 turn num_workers 8 to 0, 改了就可以run
        print("Finish turn into coreset dataset")# 0123 delete later
        
        hg_encoding = None

        # Part 1: Obtain embeddings
        # Disable autograd to speed up inference
        with torch.no_grad():
            for images in tqdm(coreset_dataloader):
                
                images = images.to(non_blocking=True, device='cuda')
                _, hourglass_features = self.hg_network(images)

                try:
                    hg_encoding = torch.cat((hg_encoding, hourglass_features['penultimate'].cpu()), dim=0)
                except TypeError:
                    hg_encoding = hourglass_features['penultimate'].cpu()

        hg_final_encoding = hg_encoding.squeeze().numpy()
        print("Finsh complete Core-Set encoding")
        print("Check hg_encoding type: ", hg_final_encoding.shape, type(hg_final_encoding))
        np.save(file=self.conf.model_save_path.format('hg_encoding.npy'), arr=hg_final_encoding) #0125
        
        logging.info('Core-Set encodings computed.')

        # Part 2: k-Centre Greedy
        ######### mixture ##########
        if mode =='core_ll':
            core_set_budget = self.conf.active_learning_params['num_images']['total']*2
        else:
            core_set_budget = self.conf.active_learning_params['num_images']['total']

        #core_set_budget = self.conf.active_learning_params['num_images']['total']
        min_distances = None

        if len(annotated_idx) != 0:
            min_distances = update_distances(cluster_centers=annotated_idx, encoding=hg_final_encoding, min_distances=None)

        coreset_idx =np.array([])
        for _ in tqdm(range(core_set_budget)):
            if len(annotated_idx) == 0:  # Initial choice of point
                # Initialize center with a randomly selected datapoint
                ind = np.random.choice(np.arange(hg_final_encoding.shape[0]))
            else:
                ind = np.argmax(min_distances)
            
                #print("ind check", ind.shape, type(ind), ind) #一個: <class 'numpy.int64'> 16193

            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            min_distances = update_distances(cluster_centers=[ind], encoding=hg_final_encoding, min_distances=min_distances)

            coreset_idx = np.concatenate([coreset_idx, [ind]], axis=0).astype(np.int32) #0123 # 0428
            #annotated_idx = np.concatenate([annotated_idx, [ind]], axis=0).astype(np.int32) #0123 original

            #print("annotated_idx check", annotated_idx.shape, type(annotated_idx), annotated_idx)

        #unlabelled_idx = np.array(list(set(train['pick_index'])-set(annotated_idx)))
        print("# of coreset select", len(coreset_idx))
        print("# of previous select", len(annotated_idx))
        annotated_idx = np.concatenate([annotated_idx, coreset_idx], axis=0).astype(np.int32)
        print("# of training data to the model(Final):", len(annotated_idx))
        np.save(file=self.conf.model_save_path.format('annotation.npy'), arr=annotated_idx)
        

        print("-------------------Coreset Method Finish-------------------------")

        return annotated_idx, coreset_idx

    def get_preds_s(self, hm, return_conf=False):
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
    def learning_loss_sampling(self, train, dataset_size, mode):
    #def learning_loss_sampling(self, train, dataset_size, hg_depth=4, mode):
        '''
        Learning loss sampling of images from training dataset
        :param train: (dict) Training dataset
        :param dataset_size: (dict) Stores the size of each dataset - MPII / LSP+LSPET
        :return: (np.ndarray) Indices chosen for sampling
        '''
        idx_2D_to_3D_s = [16, 3, 4, 5, 2, 1, 0, 7, 6, 8, 9, 12, 11, 10, 13, 14, 15]
        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model_load_path_3D, 'model_checkpoints/annotation.npy'))

        if self.conf.model_load_3D_model:
            annotated_idx = np.load(os.path.join(self.conf.model_load_path_3D, 'model_checkpoints/annotation.npy'))
        else:
            annotated_idx = np.array([])

        print("----------Learning Loss approach begin----------")

        print("Previous # of annotation is: ", len(annotated_idx))
        # Set of indices not annotated
        unlabelled_idx = np.array(list(set(train['pick_index'])-set(annotated_idx)))
        print("# of unlabelled_idx: ", len(unlabelled_idx))
        #unlabelled_dataset = {k : [val for i, val in enumerate(v) if i in unlabelled_idx] for (k, v) in train.items()} # 10000
        unlabelled_dataset = {k : [val for i, val in enumerate(v) if i not in annotated_idx] for (k, v) in train.items()} # <10000
        print("# of unlabeled data: ", len(unlabelled_dataset['input']))
        #print('unlabelled_dataset: ', type(unlabelled_dataset), unlabelled_dataset.keys()) #dict_keys(['input', 'target', 'meta', 'reg_target', 'reg_ind', 'reg_mask', 'image', 'pick_index'])
        
        

        #dataset_ = ActiveLearningDataset(dataset_dict = unlabelled_dataset) #1229
        dataset_ = ActiveLearningDataset(dataset_dict = unlabelled_dataset)
        learnloss_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.batch_size, shuffle=False, num_workers=0)

        learnloss_pred = None

        # Prediction and concatenation of the learning loss network outputs
        print("Let's go learnloss with ", self.conf.learning_loss_obj)
        with torch.no_grad():
            for images in tqdm(learnloss_dataloader):

                images = images.to(non_blocking=True, device='cuda')

                _, hourglass_features = self.hg_network(images)
                ####### 3D pred and encoding #######
                _ = _.cpu().detach().numpy().mean(axis=1)
                _max = self.get_preds_s(_)
                _max = _max.astype(np.float32)
                _max_train = []

                for i in range(_max.shape[0]):
                    b_torse = (_max[i][2]+ _max[i][3])/2
                    _max_train.append(np.concatenate((_max[i], [b_torse]), axis=0)[idx_2D_to_3D_s])

                _max_train = np.array(_max_train)
                _max_train = torch.tensor(_max_train).to('cuda')
                _max_train = _max_train/64
                _max_train = _max_train.view(_max_train.size(0), -1)
                _output, _encoding_3D = self.network_3D(_max_train)
                ##################################

                if self.conf.learning_loss_obj=='2D':
                    # # encodings = torch.cat([hourglass_features[depth] for depth in range(1, hg_depth + 2)], dim=-1)
                    # encodings = hourglass_features['penultimate']

                    # No longer concatenating, will now combine features through convolutional layers
                    encodings = torch.cat(
                        [hourglass_features['feature_5'].reshape(images.shape[0], hourglass_features['feature_5'].shape[1], -1),
                         hourglass_features['feature_4'].reshape(images.shape[0], hourglass_features['feature_4'].shape[1], -1),
                         hourglass_features['feature_3'].reshape(images.shape[0], hourglass_features['feature_3'].shape[1], -1),
                         hourglass_features['feature_2'].reshape(images.shape[0], hourglass_features['feature_2'].shape[1], -1),
                         hourglass_features['feature_1'].reshape(images.shape[0], hourglass_features['feature_1'].shape[1], -1)], dim=2)
                    learnloss_pred_ = self.learnloss_network(encodings)

                if self.conf.learning_loss_obj == '3D':
                    encodings = torch.cat((_encoding_3D[0], _encoding_3D[1]), dim=1)
                    learnloss_pred_ = self.learnloss_network(encodings)

                if self.conf.learning_loss_obj == '2D+3D':
                    
                    encodings_2D = torch.cat(
                        [hourglass_features['feature_5'].reshape(images.shape[0], hourglass_features['feature_5'].shape[1], -1),
                         hourglass_features['feature_4'].reshape(images.shape[0], hourglass_features['feature_4'].shape[1], -1),
                         hourglass_features['feature_3'].reshape(images.shape[0], hourglass_features['feature_3'].shape[1], -1),
                         hourglass_features['feature_2'].reshape(images.shape[0], hourglass_features['feature_2'].shape[1], -1),
                         hourglass_features['feature_1'].reshape(images.shape[0], hourglass_features['feature_1'].shape[1], -1)], dim=2)

                    encodings_3D = torch.cat((_encoding_3D[0], _encoding_3D[1]), dim=1)
                    learnloss_pred_ = self.learnloss_network(encodings_2D, encodings_3D)
                    # concate 再一起並且丟進learnloss_network裡面

                #learnloss_pred_, _ = self.learnloss_network(encodings)
                #learnloss_pred_ = self.learnloss_network(encodings)
                ############## 3D loss prediction #####################
                # _, learnloss_pred = self.learnloss_network(encodings)
                #######################################################
                learnloss_pred_ = learnloss_pred_.squeeze()

                try:
                    learnloss_pred = torch.cat([learnloss_pred, learnloss_pred_.cpu()], dim=0)
                except TypeError:
                    learnloss_pred = learnloss_pred_.cpu()

        # argsort defaults to ascending
        
        print("# of learnloss_pred", len(learnloss_pred))
        print("# of unlabeled indx", len(unlabelled_idx))

        pred_with_index = np.concatenate([learnloss_pred.numpy().reshape(-1, 1),
                                          unlabelled_idx.reshape(-1, 1)], axis=-1)

        pred_with_index = pred_with_index[pred_with_index[:, 0].argsort()]
        # 0513 save the learnloss prediction
        np.save(file=self.conf.model_save_path.format('pred_with_index_{}.npy'.format(self.conf.learning_loss_obj)), arr=pred_with_index)
        # 0513 save the learnloss prediciton
        
        if mode == 'll_core':
            indices = pred_with_index[-(self.conf.active_learning_params['num_images']['total']*2):, 1]
        else:
            indices = pred_with_index[-self.conf.active_learning_params['num_images']['total']:, 1]
        #indices = pred_with_index[-self.conf.active_learning_params['num_images']['total']:, 1]
        annotated_idx = np.concatenate([annotated_idx, indices], axis=0).astype(np.int32)

        # unique, counts = np.unique(train['dataset'][annotated_idx], return_counts=True)
        # proportion = {key: value for (key, value) in zip(unique, counts)}
        # with open(self.conf.model_save_path.format('sampling_proportion.txt'), "x") as file:
        #     file.write('Learning Loss sampling\n')
        #     [file.write("{}: {}\n".format(key, proportion[key])) for key in proportion.keys()]

        np.save(file=self.conf.model_save_path.format('annotation.npy'), arr=annotated_idx)
        print("-------------------LearnLoss Method Finish-------------------------")
        return annotated_idx, indices

    ############## 0518 ave ##################
    # def 
    ##########################################

    def mixture(self, train, dataset_size, mode):
        """
        ave: coreset 算一次, learning_loss 算一次
        core_ll: coreset 先挑, 從coreset裡面挑
        ll_core: learning_loss 先挑, 從learning_loss裡面挑
        """
        print("-------------------Mixture Method-------------------------")
        if self.conf.model_load_3D_model: #model_load_hg 0312
            annotated_idx = np.load(os.path.join(self.conf.model_load_path_3D, 'model_checkpoints/annotation.npy'))
        else:
            annotated_idx = np.array([])

        if mode == 'ave':
            coreset_idx = self.coreset_sampling(train, dataset_size, mode)
            ll_idx = self.learning_loss_sampling(train, dataset_size, mode)

        if mode == 'core_ll':
            # annotated_idx, coreset_idx
            # coreset_sampling跟learning_loss_sampling都要改total number
            # goto_learn_loss 要call annotation.npy 再從remain_train(4000)挑選
            # coreset_sampling: return annotated_idx(), coreset_idx
            _, coreset_idx = self.coreset_sampling(train, dataset_size, mode)

            # 0516
            final_idx = self.goto_learning_loss(train, coreset_idx)
            # 0516
            # remain_train = {k : [val for i, val in enumerate(v) if i in coreset_idx] for (k, v) in train.items()}
            # final_idx = self.goto_learning_loss(remain_train, coreset_idx)
            print("# mixture method to pick: ", len(final_idx))

        if mode == 'll_core':
            _, ll_idx = self.learning_loss_sampling(train, dataset_size, mode)
            
            final_idx = self.goto_coreset(train, ll_idx, mode)
            print("# mixture method to pick: ", len(final_idx))
             
        annotated_idx = np.concatenate([annotated_idx, final_idx], axis=0).astype(np.int32)
        np.save(file=self.conf.model_save_path.format('annotation.npy'), arr=annotated_idx)
        print("-------------------Mixture Method Finish-------------------------")

        return annotated_idx, 0

        
    def expected_gradient_length_sampling(self, train, dataset_size):
        """
        https://arxiv.org/abs/2104.09493
        """
        raise NotImplementedError('The proposed Expected Gradient Length (EGL++) method is currently under review.')


    def multipeak_entropy(self, train, dataset_size, mode):
        '''
        Multi-peak entropy sampling of images from training dataset
        :param train: (dict) Training dataset
        :param dataset_size: (dict) Stores the size of each dataset - MPII / LSP+LSPET
        :return: (np.ndarray) Indices chosen for sampling
        '''
        print("MPE method")
        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model_load_path, 'model_checkpoints/annotation.npy'))

        if self.conf.model_load_3D_model: #model_load_hg 0312
            annotated_idx = np.load(os.path.join(self.conf.model_load_path_3D, 'model_checkpoints/annotation.npy'))
        else:
            annotated_idx = np.array([])

        #unlabelled_idx = np.array(list(set(train['index']) - set(annotated_idx))) #0312

        unlabelled_idx = np.array(list(set(train['pick_index']) - set(annotated_idx)))

        # Multi-peak entropy only over the unlabelled set of images
        #dataset_ = ActiveLearningDataset(dataset_dict=train, indices=unlabelled_idx)

        # turn unlabelled_idx into new_dataset_ 0426
        remain_train = {k : [val for i, val in enumerate(v) if i in unlabelled_idx] for (k, v) in train.items()}
        dataset_ = ActiveLearningDataset(dataset_dict=remain_train)

        mpe_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.batch_size, shuffle=False,
                                                     num_workers=8)

        hg_heatmaps = None

        # Part 1: Obtain set of heatmaps
        # Disable autograd to speed up inference
        print("start with multi-entropy, inference and obtain set of heatmaps")
        with torch.no_grad():
            for images in tqdm(mpe_dataloader):

                images = images.to(non_blocking=True, device='cuda')
                hg_heatmaps_, _ = self.hg_network(images)

                try:
                    hg_heatmaps = torch.cat((hg_heatmaps, hg_heatmaps_[:, -1, :, :, :].cpu()), dim=0)
                except TypeError:
                    hg_heatmaps = hg_heatmaps_[:, -1, :, :, :].cpu()

        hg_final_heatmaps = hg_heatmaps.squeeze().numpy()
        logging.info('Multi-peak entropy heatmaps computed.')


        # Part 2: Multi-peak entropy
        mpe_budget = self.conf.active_learning_params['num_images']['total']
        mpe_value_per_img = np.zeros(hg_final_heatmaps.shape[0], dtype=np.float32)

        # e.g. shape of heatmap final is BS x 14 x 64 x 64
        for i in tqdm(range(hg_final_heatmaps.shape[0])):
            normalizer = 0
            entropy = 0
            for hm in range(hg_final_heatmaps.shape[1]):
                loc = peak_local_max(hg_final_heatmaps[i, hm], min_distance=7, threshold_abs=7.5)
                peaks = hg_final_heatmaps[i, hm][loc[:, 0], loc[:, 1]]

                if peaks.shape[0] > 0:
                    normalizer += 1
                    peaks = softmax_fn(peaks)
                    entropy += entropy_fn(peaks)

            mpe_value_per_img[i] = entropy

        mpe_value_per_img = torch.from_numpy(mpe_value_per_img)
        print("after mpe_value_per_img: ", len(mpe_value_per_img))
        vals, idx = torch.topk(mpe_value_per_img, k=mpe_budget, sorted=False, largest=True)
        assert idx.dim() == 1, "'idx' should be a single dimensional array"
        print("annotated idx", len(annotated_idx))
        print("idx", len(idx.numpy()))

        
        annotated_idx = np.concatenate([annotated_idx, unlabelled_idx[idx.numpy()]], axis=0).astype(np.int32)

        #0312: unique, counts = np.unique(train['dataset'][annotated_idx], return_counts=True), KeyError: 'dataset'
        #unique, counts = np.unique(train['dataset'][annotated_idx], return_counts=True)
        #proportion = {key: value for (key, value) in zip(unique, counts)}


        print("Done with multi-entropy, saving....")
        # with open(self.conf.model_save_path.format('sampling_proportion.txt'), "x") as file:
        #     file.write('MPE sampling\n')
        #     [file.write("{}: {}\n".format(key, proportion[key])) for key in proportion.keys()]

        np.save(file=self.conf.model_save_path.format('annotation.npy'), arr=annotated_idx)

        # 0212 not really sure
        return annotated_idx, 0

class ActiveLearningDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dict, indices=None):
        '''
        Helper class to initialize Dataset for torch Dataloader
        :param dataset_dict: (dict) Containing the dataset in numpy format
        :param indices: (np.ndarray) Which indices to use for generating the dataset
        '''
        self.len = len(dataset_dict['input'])

        #len(unlabelled_dataset['input'])
        if indices is None:
            self.inp = dataset_dict['input']
            # self.bounding_box = dataset_dict['bbox_coords']

        else:
            self.inp = dataset_dict['input'][indices]
            #self.images = dataset_dict['img'][indices]
            #self.bounding_box = dataset_dict['bbox_coords'][indices]

        # self.xy_to_uv = lambda xy: (xy[1], xy[0])

    def __len__(self):
        # return self.images.shape[0]
        return self.len

    def __getitem__(self, item):
        '''

        :param item:
        :return:
        '''
        inp = self.inp[item]
        inp = np.transpose(inp, (1, 2, 0))
        #inp = inp.permute(0, 2, 3, 1)
        # image =  self.images[item]
        # bounding_box = self.bounding_box[item]

        # Determine crop
        # img_shape = np.array(image.shape)

        # # Bounding box for the first person
        # [min_x, min_y, max_x, max_y] = bounding_box[0]

        # tl_uv = self.xy_to_uv(np.array([min_x, min_y]))
        # br_uv = self.xy_to_uv(np.array([max_x, max_y]))
        # min_u = tl_uv[0]
        # min_v = tl_uv[1]
        # max_u = br_uv[0]
        # max_v = br_uv[1]

        # centre = np.array([(min_u + max_u) / 2, (min_v + max_v) / 2])
        # height = max_u - min_u
        # width = max_v - min_v

        # scale = 2.0

        # top_left = np.array([centre[0] - (scale * height / 2), centre[1] - (scale * width / 2)])
        # bottom_right = np.array([centre[0] + (scale * height / 2), centre[1] + (scale * width / 2)])

        # top_left = np.maximum(np.array([0, 0], dtype=np.int16), top_left.astype(np.int16))
        # bottom_right = np.minimum(img_shape.astype(np.int16)[:-1], bottom_right.astype(np.int16))

        # # Cropping the image
        # image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]

        # # Resize the image
        # image = self.resize_image(image, target_size=[256, 256, 3])

        return inp
        # return torch.tensor(data=image / 256.0, dtype=torch.float32, device='cpu')

    # def resize_image(self, image_=None, target_size=None):
    #     '''

    #     :return:
    #     '''
    #     # Compute the aspect ratios
    #     image_aspect_ratio = image_.shape[0] / image_.shape[1]
    #     tgt_aspect_ratio = target_size[0] / target_size[1]

    #     # Compare the original and target aspect ratio
    #     if image_aspect_ratio > tgt_aspect_ratio:
    #         # If target aspect ratio is smaller, scale the first dim
    #         scale_factor = target_size[0] / image_.shape[0]
    #     else:
    #         # If target aspect ratio is bigger or equal, scale the second dim
    #         scale_factor = target_size[1] / image_.shape[1]

    #     # Compute the padding to fit the target size
    #     pad_u = (target_size[0] - int(image_.shape[0] * scale_factor))
    #     pad_v = (target_size[1] - int(image_.shape[1] * scale_factor))

    #     output_img = np.zeros(target_size, dtype=image_.dtype)

    #     # Write scaled size in reverse order because opencv resize
    #     scaled_size = (int(image_.shape[1] * scale_factor), int(image_.shape[0] * scale_factor))

    #     padding_u = int(pad_u / 2)
    #     padding_v = int(pad_v / 2)

    #     im_scaled = cv2.resize(image_, scaled_size)
    #     # logging.debug('Scaled, pre-padding size: {}'.format(im_scaled.shape))

    #     output_img[padding_u : im_scaled.shape[0] + padding_u,
    #                padding_v : im_scaled.shape[1] + padding_v, :] = im_scaled

    #     return output_img
