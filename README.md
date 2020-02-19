SSN
-

Introduction
-

SSN is the abstraction of Structured Segment Network. SSN can be used to do action detection. Given a video, it can detect the actions contained in the video and tell the class, begin time and end time of these actions. The papar where SSN comes from is [Temporal Action Detection with Structured Segment Networks](http://arxiv.org/abs/1704.06228). The original code of SSN is [yjxiong/action-detection](https://github.com/yjxiong/action-detection). The code in this repository is basically based on [yjxiong/action-detection](https://github.com/yjxiong/action-detection).

The general steps of deep learning are dataset, which is creating a structure to represent your data, model, which is a thing that is used to fit the data, train and test, train is making the model fit the data, test is using the model on unseen data and get the result.

In SSN, dataset is a hierarchical thing, first you can get a video, then you can get every proposal of the video, video is a data unit, it has many properties and functions, including its proposals, proposal is also a data unit, it also has many properties and functions, model is basically a base model like BNInception or InceptionV3, then there are three linear layers appending to the end of the base model to compute the activity score, completeness score and regression score seperately, there is also a STPP layer between the base model and the linear layers, its function is to strength the model's ability to fit the data, train is finding a number which you want to get smaller, activity loss, completeness loss, regression loss are picked, activity loss is about whether the class of the action is correct, completeness loss is about whether the action is complete, regression loss is about whether the location and length of the action are correct.

Dependency
-

- Ubuntu 
- torch 1.4.0
- torchvision 0.5.0
- Pillow 7.0.0
- GPU

How To Use
-

1. Download this repository.

        git clone --recursive https://github.com/Ruiyang-061X/SSN.git

2. Download thumos14 from [here](https://www.crcv.ucf.edu/THUMOS14/download.html). Note that the validationset of thumos14 is the trainset of SSN and the testset of thumos14 is the validationset of SSN.

3. Prepare the dataset. Please follow the instructions in [yjxiong/action-detection/README.md](https://github.com/yjxiong/action-detection/blob/master/README.md). Follow the instructions in [Extract Frames and Optical Flow Images](https://github.com/yjxiong/action-detection#extract-frames-and-optical-flow-images) to prepare the frames. Follow the instructions in [Prepare the Proposal Lists](https://github.com/yjxiong/action-detection#prepare-the-proposal-lists) to prepare the proposal lists. You should put thumos14_tag_val_proposal_list.txt and thumos14_tag_test_proposal_list.txt in `dataset/`.

4. Excute the following command to train the model. The trained models are saved in `trained_model/`

        python3 train.py --modality MODALITY
    
    MODALITY can be RGB, RGBDiff or Flow.

Result
-

Due to lack of computation resource, the model is left untrained.