# AL_3D_Pose

## Overview
Three folder
- code
- exp: place the 3D model
- data
## Dataset

- Download h36m_cache_train.npy(for training)
    - h36m_cache_train_15000.npy
    - h36m_cache_train_15001.npy
    - h36m_cache_train_15002.npy
    - 放在code/
    - ![](https://i.imgur.com/6rZ9bBY.png)
- Download h36m dataset (for testing)
    - 放在data/
    - ![](https://i.imgur.com/IBnJdyY.png)

## Configuration_3D.yml
- 改exp_name
- train/test_simple_model 擇一為true
- 確認load_model_path

## Train
- Setting ```configuration_3D.yml```
    - Load 2D model: ```Best_2D_model``` or ```Train_h36m``` 
    - Load 3D model: ```exp/0615_h36m_Train_h36m_1```
    - run ```train.py```
## Inference
- Setting ```configuration_3D.yml```
    - Load 2D model: ```Best_2D_model``` or ```Train_h36m``` 
    - Load 3D model: ``exp/0615_h36m_Train_h36m_1```
- run ```eval.py```
