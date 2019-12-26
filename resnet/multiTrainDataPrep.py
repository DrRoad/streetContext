#this code will prepare the train test split for the data which should be stored in a single directory

import pandas as pd
import dataset
#from dataset import ImageFolderWithPaths
from torchvision import datasets, transforms
import torch
from matplotlib import pylab as plt
import numpy as np
from collections import Counter
import os


#%%
def trainOrTest(row,sampleDataset):
    if row['fileName'] in sampleDataset:
        return 'Test'
    else:
        return 'Train'

#%%
def GetFiles():
    #SF data paths
    data_dir = '/home/fahad/data/streetContext/streetImages/SF/modelRelated/data/resnet/cleaning/train'

    train_dataset = datasets.ImageFolder(root=data_dir)
    classes = list(train_dataset.class_to_idx)

    base_transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolderWithPaths(root=data_dir, transform=base_transform)  # our custom dataset
    dataloader = torch.utils.data.DataLoader(dataset)
    classes = list(train_dataset.class_to_idx)

    files=[ [label, path] for input, label, path in dataloader]

    #write files file
    with open('/home/fahad/data/streetContext/streetImages/SF/modelRelated/data/imagesTracker.csv','w') as f:
        f.writelines("classification,fileName,filePath\n")
        for i in files:
            f.writelines("{},{},{}\n".format(classes[i[0].numpy()[0]],i[1][0].split("/")[-1].replace("\n",''),i[1][0]))
    return True

def GenerateManySplits():
    GetFiles()
    filePath='/home/fahad/data/streetContext/streetImages/SF/modelRelated/data/imagesTracker.csv'

    pandasFile=pd.read_csv(filePath)


    with open(filePath,'r') as infile:
        infile.readline()
        fileNamesList=[line.replace('\n','').split(',')[1] for line in infile.readlines()]

    dataSize=len(fileNamesList)
    testSetSize=3000;

    trainIterations=50

    for i in range(trainIterations):
        #getting a random sample
        sample=np.random.choice(dataSize,testSetSize,replace=False)
        sampleDataset=[fileNamesList[i] for i in sample]
        print(sampleDataset)
        pandasFile[str(i)]=pandasFile.apply(lambda row: trainOrTest(row,sampleDataset),axis=1)

    pandasFile.to_csv('/home/fahad/data/streetContext/streetImages/SF/modelRelated/data/imagesTrackerTrainTest.csv')

#%%
def GetFilesToExclude(filePath,outPath,i):
    data = pd.read_csv(filePath)

    def count(row):
        total = correct = 0;
        for i in range(5, 50):
            if row[str(i)] == "Test":
                total += 1;
                if row[str(i) + "_result"] == True:
                    correct += 1
        return tuple([correct, total])

    data['metrics'] = data.apply(lambda x: count(x), axis=1)
    data['correct'] = data.apply(lambda x: x['metrics'][0], axis=1)
    data['total'] = data.apply(lambda x: x['metrics'][1], axis=1)

    #plt.hist(data[data['total'] > i]['correct'], bins=np.linspace(0, 20, 20))
    #plt.show()
    #print(Counter(data[data['total'] > i]['correct']))
    #plt.hist(data['total'], bins=np.linspace(0, 20, 20))
    #plt.show()
    #print(Counter(data[data['total'] > i]['correct']))
    # get all images with at least 11 evaluations and never been correctly classified

    thDataTotal = data[data['total'] > i]
    thDataTotalCount = thDataTotal[thDataTotal['correct'] == 0]
    fileNames = thDataTotalCount['filePath']

    fileNames.to_csv(outPath+"filesToExclude.csv",index=False,header='filePath')

#%%
def move(row):
    moveCommand='mv \'{}\' \'{}\''.format(row['filePath'],row['filePath'].replace('train','train_m'))
    os.system(moveCommand)
    pass

#%% this cell reads images folder and sets up the train test splitting for many iterations
GenerateManySplits()

#%% this cell is for the cleaning
filePath="/home/fahad/data/streetContext/streetImages/SF/modelRelated/logfiles/resnet18_cleaning/imagesTrackerStats_49.csv"
outPath="/home/fahad/data/streetContext/streetImages/SF/modelRelated/logfiles/resnet18_cleaning/"
GetFilesToExclude(filePath,outPath,16)
#%% move excluded files to train_m
excludedFilesList="/home/fahad/data/streetContext/streetImages/SF/modelRelated/logfiles/resnet18_cleaning/filesToExclude.csv"
data=pd.read_csv(excludedFilesList)
data.apply(lambda x: move(x), axis=1)

