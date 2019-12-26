#this is the vanilla train code
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
import dataset
from models.AlexNet import *
from models.ResNet import *
import csv
from os.path import expanduser
from ClassificationCleaning import clean
from importlib import reload
import torchvision




training_loss=[];
validation_loss=[];
logid = np.random.random()

home = expanduser("~")

#SF Paths configuration
#data_root = home+'/data/streetContext/streetImages/SF/modelRelated/data/resnetPOI/'
#modelsPath=home+'/data/streetContext/streetImages/SF/modelRelated/models/Alexnet_poi/';
#log_folder_path=home+'/data/streetContext/streetImages/SF/modelRelated/logfiles/Alexnet_poi/';

#Boston Paths configuration
data_root = home+'/data/streetContext/streetImages/Boston/modelRelated/data/resnetPOI/' 
modelsPath=home+'/data/streetContext/streetImages/Boston/modelRelated/models/resnet18/'
log_folder_path=home+'/data/streetContext/streetImages/Boston/modelRelated/logfiles/resnet18/'

def writeArrayToCsv(array,path):
    with open(path,'w') as outfile:
      wr=csv.writer(outfile)
      wr.writerow(array)
      outfile.close()

def write2dArraytoCsv(array,path):
    with open(path,'w') as outfile:
      wr=csv.writer(outfile)
      wr.writerows(array)
      outfile.close()


def getConfusionMatrix(labels, predictions,nClasses=10):
        confusionMatrix=np.zeros([nClasses,nClasses])
        correct=(labels==predictions).sum().item()

        wrong=len(labels)-correct;
        outOfRange=0;
        for i in range(len(labels)):
            if predictions[i] < nClasses:
                confusionMatrix[labels[i],predictions[i]]+=1;
            else:
                outOfRange+=1;

        return wrong,correct,outOfRange,confusionMatrix

def validation(val_loader,model,device,ii,nClasses):

    setSize=loss=i=0;
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_num, (val_inputs, val_labels) in enumerate(val_loader, 1):

        gc.collect()
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)
        val_outputs = model(val_inputs)
        val_predictions = torch.argmax(val_outputs, 1)
        loss += criterion(val_outputs, val_labels).item()
        setSize+=len(val_labels)

        if i==0:
            wrong, correct, outOfRange, confusionMatrix=getConfusionMatrix(val_labels, val_predictions, nClasses)
        else:
            a1,a2,a3,a4 = getConfusionMatrix(val_labels, val_predictions, nClasses)
            wrong=wrong+a1
            correct=correct+a2
            outOfRange=outOfRange+a3
            confusionMatrix=confusionMatrix+a4
        i+=1;
    validation_loss.append(loss/setSize)
    writeArrayToCsv(validation_loss,log_folder_path+'validation_loss'+str(ii)+'.csv')

    #print(confusionMatrix)
    write2dArraytoCsv(confusionMatrix,log_folder_path+'confusion_matrix'+str(ii)+'.csv')
    print("-------------------------------- with {} wrong and {} correct and {} outOfRange \n--Classification accuracy {}".format(wrong, correct,
                                                                                                   outOfRange,correct/(wrong+correct+outOfRange)))

    return correct/(wrong+correct+outOfRange)

def run(ii,startFromPath='None'):
    # Parameters
    num_epochs = 2000#15 + ii*7
    output_period = 50
    batch_size = 50


    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nClasses=10
    #model = torchvision.models.inception_v3(num_classes=11,aux_logits=False)
    model= resnet_18() #resnet_18()  # alexnet()
            #torchvision.models.ince
    #this allows for the model to proceed with training after the starting point
    if startFromPath!='None':
        model.load_state_dict(torch.load(startFromPath))
        model.eval()


    model = model.to(device)
    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    logid = np.random.random()
    num_train_batches = len(train_loader)

    criterion = nn.CrossEntropyLoss().to(device)
    # TODO: optimizer is currently unoptimized
    # there's a lot of room for improvement/different optimizers
    #optimizer = optim.SGD(model.parameters(), lr=1e-4)
    LEARNING_RATE=5e-6
    #inception_LR=1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=0)


    epoch = 1
    while epoch <= num_epochs:

        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))

        model.train()
        lossTot=imageSetSize=0;

        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            predictions=torch.argmax(outputs,1)
            #getConfusionMatrix(labels, predictions, 10)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

#            print('[%d:%.2f] loss: %.3f' % (
#                epoch, batch_num*1.0/num_train_batches,
#                l oss.item()/len(labels)
#                ))

            gc.collect()
            imageSetSize += len(labels)
            lossTot+=loss.item();
        gc.collect()
        print("epoch {}".format(epoch))
        training_loss.append(lossTot/imageSetSize)
        val_accuracy=validation(val_loader,model,device,ii,nClasses)
        writeArrayToCsv(training_loss, log_folder_path + 'training_loss'+str(ii)+'.csv')
        
        if val_accuracy>0.8:
        # save after every epoch
               print('writing this model at epoch {} and accuracy {}'.format(epoch,val_accuracy))
               torch.save(model.state_dict(), modelsPath+"/BOS_POI_resnet18.%d" % epoch)

        # TODO: Calculate classification error and Top-5 Error
        # on training and validation datasets here

        gc.collect()
        epoch += 1


def main():
    for i in range(1):
        print('Starting training {}'.format(i))

        #start from and end training point
        #run(i,startFromPath='/home/fahad/streetContexts/data/streetImages/Boston/modelRelated/models/inceptionv3_1/inceptionv3_model.4999')

        run(i)
        #print('cleaning')
        #print('Training terminated')
        #clean("train/","train_m/","missClass",i)
        #clean("train_m/","train/","correctClass",i)
        #print("cleaning terminated")


main()
