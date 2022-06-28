
# dataloader for mpi_inf_3dhp dataset
# - [X] main_ver2.py 加入mpi_inf_3dhp dataset: from mpi_inf_3dhp_dataloader import mpi_inf_3dhp_AL
# - [ ] 確認 mpi_inf_3dhp data structure

# train/val/eval 
# total number/position_2D/position_3D/joint_num 才能知道get_item 要放什麼資訊
# annotation_path/img_path/preprocessing_info

# load_h36m(create h36m dict): mpi_inf_3dhp_dict 

# - [ ] 改mpi_inf_3dhp_AL 裡面的東西

class mpi_inf_3dhp_AL(torch.utils.data.Dataset):
    '''
    inference pick indices:
    (1) self.indices, self.pick_indices
    (2) 
    '''
    def __init__(self, mpi_inf_3dhp_dict, activelearning_obj, getitem_dump, conf, **kwargs): #0224_AL_Dataset
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


            
            print("Turn into new dataset")
            print("0516 check", len(self.indices))
            print("0516 check train['input']", len(self.train_entire['pick_index']))

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
