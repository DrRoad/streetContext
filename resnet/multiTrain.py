#this file will take in many splits of data and do independent but many train-test splits
#the output for each run is stored in a csv file

import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.adam as adam
import torch.optim.lr_scheduler
from dataset import ImageFolderWithPaths
torch.backends.cudnn.benchmark = True
import numpy as np
import matplotlib.pyplot as plt
import dataset
from models.AlexNet import *
from models.ResNet import *
import csv
from os.path import expanduser
from ClassificationCleaning import clean
from importlib import reload
import torchvision
import pandas as pd
import os
from torchvision import datasets, transforms


training_loss = [];
validation_loss = [];
logid = np.random.random()

home = expanduser("~")

data_root = home + '/data/streetContext/streetImages/SF/modelRelated/data/resnet/cleaning'
modelsPath = home + '/data/streetContext/streetImages/SF/modelRelated/models/resnet18_cleaning/';
log_folder_path = home + '/data/streetContext/streetImages/SF/modelRelated/logfiles/resnet18_cleaning/';


def writeArrayToCsv(array, path):
    with open(path, 'w') as outfile:
        wr = csv.writer(outfile)
        wr.writerow(array)
        outfile.close()


def write2dArraytoCsv(array, path):
    with open(path, 'w') as outfile:
        wr = csv.writer(outfile)
        wr.writerows(array)
        outfile.close()


def getConfusionMatrix(labels, predictions, nClasses=10):
    confusionMatrix = np.zeros([nClasses, nClasses])
    correct = (labels == predictions).sum().item()

    wrong = len(labels) - correct;
    outOfRange = 0;
    for i in range(len(labels)):
        if predictions[i] < nClasses:
            confusionMatrix[labels[i], predictions[i]] += 1;
        else:
            outOfRange += 1;

    return wrong, correct, outOfRange, confusionMatrix


def validation(val_loader, model, device, ii, nClasses,epoch):
    logfile = open(log_folder_path + "screenLog.txt", "a+")
    setSize = loss = i = 0;
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_num, (val_inputs, val_labels) in enumerate(val_loader, 1):

        gc.collect()
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)
        val_outputs = model(val_inputs)
        val_predictions = torch.argmax(val_outputs, 1)
        loss += criterion(val_outputs, val_labels).item()
        setSize += len(val_labels)

        if i == 0:
            wrong, correct, outOfRange, confusionMatrix = getConfusionMatrix(val_labels, val_predictions, nClasses)
        else:
            a1, a2, a3, a4 = getConfusionMatrix(val_labels, val_predictions, nClasses)
            wrong = wrong + a1
            correct = correct + a2
            outOfRange = outOfRange + a3
            confusionMatrix = confusionMatrix + a4
        i += 1;
    validation_loss.append(loss / setSize)
    writeArrayToCsv(validation_loss, log_folder_path + 'validation_loss' + str(ii) + '.csv')

    # print(confusionMatrix)
    write2dArraytoCsv(confusionMatrix, log_folder_path + 'confusion_matrix' + str(ii) + '.csv')
    msg="epoch {}-{}-------------------------------- with {} wrong and {} correct and {} outOfRange \n--Classification accuracy {} \n".format(ii,epoch,wrong, correct,outOfRange, correct / (wrong + correct + outOfRange))
    print(msg)
    logfile.writelines(msg)
    logfile.close()

def run(ii, startFromPath='None'):
    # Parameters
    num_epochs = 150  # 15 + ii*7
    output_period = 50
    batch_size = 50

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nClasses = 11
    # model = torchvision.models.inception_v3(num_classes=11,aux_logits=False)
    model = resnet_18()  # alexnet()# resnet_34()
    # torchvision.models.ince
    # this allows for the model to proceed with training after the starting point
    if startFromPath != 'None':
        model.load_state_dict(torch.load(startFromPath))
        model.eval()

    model = model.to(device)
    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    logid = np.random.random()
    num_train_batches = len(train_loader)

    criterion = nn.CrossEntropyLoss().to(device)
    # TODO: optimizer is currently unoptimized
    # there's a lot of room for improvement/different optimizers
    # optimizer = optim.SGD(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
    print("log available at {}".format(log_folder_path+"screenLog.txt"))


    epoch = 1
    while epoch <= num_epochs:

        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))

        model.train()
        lossTot = imageSetSize = 0;

        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            predictions = torch.argmax(outputs, 1)
            # getConfusionMatrix(labels, predictions, 10)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #            print('[%d:%.2f] loss: %.3f' % (
            #                epoch, batch_num*1.0/num_train_batches,
            #                l oss.item()/len(labels)
            #                ))

            gc.collect()
            imageSetSize += len(labels)
            lossTot += loss.item();
        gc.collect()
        print("epoch {}".format(epoch))
        training_loss.append(lossTot / imageSetSize)
        validation(val_loader, model, device, ii, nClasses,epoch)
        writeArrayToCsv(training_loss, log_folder_path + 'training_loss' + str(ii) + '.csv')

        if epoch > 100:
            # save after every epoch
            torch.save(model.state_dict(), modelsPath + "/resnet18_cleaning.%d" % epoch)

        # TODO: Calculate classification error and Top-5 Error
        # on training and validation datasets here

        gc.collect()
        epoch += 1
    return model


def move(row,dist,iteration,direction):
    #print(row[iteration])
    if direction=='toTrain':
        source=row['filePath'].replace('train','test')
    else:
        source=row['filePath']


    if row[iteration]=='Test':
        moveCommand='mv \'{}\' \'{}/{}/{}\''.format(source,dist,row['classification'],row['fileName'])

        os.system(moveCommand)
        #print(moveCommand)
    pass

def track(trainPath,valPath,ii,model,fileTracker):
    correctlyClassified=[];
    base_transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolderWithPaths(root=valPath, transform=base_transform)  # our custom dataset
    val_loader = torch.utils.data.DataLoader(dataset)

    nClasses=11
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setSize = loss = i = 0;
    criterion = nn.CrossEntropyLoss().to(device)


    for input, label, path in val_loader:

        # use the above variables freely
        input = input.to(device)
        modelOutput = model(input)
        prediction = torch.argmax(modelOutput, 1)
        label = label.to(device)

        fileName = path[0].split('/')[-1]

        if (prediction == label):
            correctlyClassified.append(fileName)

    dataset = ImageFolderWithPaths(root=valPath, transform=base_transform)  # our custom dataset
    train_loader = torch.utils.data.DataLoader(dataset)

    for input, label, path in train_loader:
        # use the above variables freely
        fileTracker['{}_result'.format(ii)]=False
        input = input.to(device)
        modelOutput = model(input)
        prediction = torch.argmax(modelOutput, 1)
        label = label.to(device)
        fileName=path[0].split('/')[-1]


        if (prediction == label):
            correctlyClassified.append(fileName)

    return correctlyClassified


def updateTracker(row,correctlyClassified):
    if row['fileName'] in correctlyClassified:
        return True
    else:
        return False

def main():
    filePath='/home/fahad/data/streetContext/streetImages/SF/modelRelated/data/imagesTrackerTrainTest.csv'
    trainPath='/home/fahad/data/streetContext/streetImages/SF/modelRelated/data/resnet/cleaning/train/'
    valPath='/home/fahad/data/streetContext/streetImages/SF/modelRelated/data/resnet/cleaning/test/'
    fileTracker=pd.read_csv(filePath)

    for i in range(5,50):
        fileTracker.apply(lambda row: move(row,valPath,str(i),'toTest'),axis=1)

        model=run(i)
        correctlyClassified=set(track(trainPath,valPath,i,model,fileTracker))
        print('----------------')
        print("number of correctly classified {} in iteration {}".format(len(correctlyClassified),i))
        print('----------------')
        fileTracker['{}_result'.format(i)]=fileTracker.apply(lambda row: updateTracker(row,correctlyClassified),axis=1)
        fileTracker.apply(lambda row: move(row,trainPath,str(i),'toTrain'),axis=1)
        fileTracker.to_csv('/home/fahad/data/streetContext/streetImages/SF/modelRelated/data/imagesTrackerStats_{}.csv'.format(i))
main()
