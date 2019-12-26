#this code is for our first generation data
#fitting -- cleaning -- fitting -- cleaning ...etc
import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.adam as adam
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True
import numpy as np
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
import dataset
from dataset import ImageFolderWithPaths
from models.AlexNet import *
from models.ResNet import *
import csv
import torch
from torchvision import datasets, transforms
import PIL
import numpy as np
import os
from os.path import expanduser


#setting up paths

home = expanduser("~")
#SF
sModelPath=home + '/data/streetContext/streetImages/SF/modelRelated/models/resnet18_poi/'
sDataPath=home + '/data/streetContext/streetImages/SF/modelRelated/data/resnetPOI/'

#Boston
#sModelPath=home + '/streetContexts/data/streetImages/Boston/modelRelated/models/resnet18/'
#sDataPath=home + '/data/streetContext/streetImages/Boston/modelRelated/data/resnetPOI/'

def clean(source,dist,classType,ii):
    # %%
    #Boston models for resnet
    #modelPath = home + '/streetContexts/data/streetImages/Boston/modelRelated/models/resnet34/resnet34_model.{}'.format(8 + ii*7)


    #SF models for resnet 18
    modelPath = sModelPath+'d_resnet18_afterCleaning.{}'.format(ii)

    print(modelPath)

    # %% read the model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_18()
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    model.to(device)
    # %%
    # EXAMPLE USAGE:

    # instantiate the dataset and dataloader

    #BOSTON data paths
    data_dir = sDataPath+source
    data_missclass_dir = sDataPath+dist

    #SF data paths
    #data_dir = home + '/data/streetContext/streetImages/SF/modelRelated/data/resnet/' + source
    #data_missclass_dir = home + '/data/streetContext/streetImages/SF/modelRelated/data/resnet/' + dist

    # %%
    train_dataset = datasets.ImageFolder(root=data_dir)
    classes = list(train_dataset.class_to_idx)

    # %%
    base_transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolderWithPaths(root=data_dir, transform=base_transform)  # our custom dataset
    dataloader = torch.utils.data.DataLoader(dataset)

    # %%
    # iterate over data
    commands = [];
    count = 0;
    for input, label, path in dataloader:
        # use the above variables freely
        input = input.to(device)
        modelOutput = model(input)
        prediction = torch.argmax(modelOutput, 1)
        label = label.to(device)
        if (prediction != label and classType=="missClass") or (prediction == label and classType=="correctClass"):

            label_str = classes[label]
            if not os.path.exists(data_missclass_dir + label_str):
                os.mkdir(data_missclass_dir + label_str)

            if "keep" not in path[0]:
                count += 1
                commands.append(r'mv "{}" "{}"'.format(path[0], data_missclass_dir + label_str + '/'))

    print("there were {} missclassified".format(count))

    # %%
    for cmd in commands:
        #print(cmd)
        #print('moved one')
        os.system(cmd)
    return True




def split(source,dist,probability):
    # %%
    home = expanduser("~")

    # %%
    # EXAMPLE USAGE:

    # instantiate the dataset and dataloader

    #BOSTON data paths
    #data_dir = home + '/streetContexts/data/streetImages/Boston/modelRelated/data/'+source
    #data_missclass_dir = home + '/streetContexts/data/streetImages/Boston/modelRelated/data/'+dist

    #SF data paths
    from_dir = sDataPath + source
    to_dir = sDataPath + dist

    # %%
    train_dataset = datasets.ImageFolder(root=from_dir)
    classes = list(train_dataset.class_to_idx)

    # %%
    base_transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolderWithPaths(root=from_dir, transform=base_transform)  # our custom dataset
    dataloader = torch.utils.data.DataLoader(dataset)

    # %%
    # iterate over data
    commands = [];
    count = 0;

    for input, label, path in dataloader:
            label_str = classes[label]
            if np.random.random()<probability:
                count+=1;
                commands.append(r'mv "{}" "{}"'.format(path[0], to_dir + label_str + '/'))

    print("Moving {} images from {} to {}".format(count,from_dir,to_dir))

    # %%
    for cmd in commands:
        #print(cmd)
        #print('moved one')
        os.system(cmd)
    return True

def main():
    #this section is for the purpose of cleaning a dataset
    for i in [999]:
        #misclassified images go to train_m
        clean('train/', 'train_m/', 'missClass',i )
        clean('test/', 'test_m/', 'missClass',i )

        clean('train_m/', 'train/', 'correctClass', i)
        clean('test_m/', 'test/', 'correctClass', i)

        #print(i)
        #correctly classified images go to train
        clean('test/', 'train/', 'correctClass', i)
        #clean('/test/', '/train/', 'missClass', i)

        #clean('cleaning/train/', 'train/', 'correctClass',i )


    split('train/','test/',0.2)

if __name__ == '__main__':
    main()

