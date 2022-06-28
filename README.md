# AL_3D_Pose

## Overview
Three folder
- code
- exp: place the 3D model
- data

![image](https://user-images.githubusercontent.com/72399747/176245611-8336d7b4-0853-4ec7-933a-abd1c6a38047.png)

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
- load 2D/3D model = True
- load_model_path
- 以下紅色地方
- ![image](https://user-images.githubusercontent.com/72399747/176245917-46fd4f09-e01c-4fe6-8843-9c2accfd3c25.png)


## Train
- Setting ```configuration_3D.yml```
    - Load 2D model: ```Best_2D_model``` or ```Train_h36m``` 
    - Load 3D model: ```exp/0615_h36m_Train_h36m_1```
    - run ```train.py```
## Inference
- Setting ```configuration_3D.yml```
    - Load 2D model: ```Best_2D_model``` or ```Train_h36m``` 
    - Load 3D model: ```exp/0615_h36m_Train_h36m_1```
- run ```eval.py```
