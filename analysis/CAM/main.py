"""
Class Activation Mapping
Googlenet, Kaggle data
"""

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os




# functions
CAM             = 1
USE_CUDA        = 1
RESUME          = 0
PRETRAINED      = 0


# hyperparameters
BATCH_SIZE      = 32
IMG_SIZE        = 224
LEARNING_RATE   = 0.0001
EPOCH           = 5


# prepare data
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

#Paths to model and data
mPath='/home/fahad/streetContext/data/streetImages/Boston/modelRelated/models/inceptionv3_1/inceptionv3_model.z849'
trainPath='/home/fahad/data/streetContext/streetImages/Boston/modelRelated/data/resnet/train/'
testingPath='/home/fahad/data/streetContext/streetImages/Boston/modelRelated/data/resnet/test/'
camOutputDir='/home/fahad/data/streetContext/streetImages/Boston/modelRelated/cam/'


train_data = datasets.ImageFolder(trainPath, transform=transform_train)
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_data = datasets.ImageFolder(testingPath, transform=transform_test)
testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
#test_dataset = datasets.ImageFolder(root=trainPath)

#getting the labels
classeslabels = list(train_data.class_to_idx)
print("this is the " ,train_data.class_to_idx)
classes={i:classeslabels[i] for i in range(0,len(classeslabels))}
# class
#classes = {0: 'cat', 1: 'dog'}

print(len(classes))
# network
net =  torchvision.models.inception_v3(num_classes=11,aux_logits=False)#inception_v3(pretrained=0)

if PRETRAINED:
    net.load_state_dict(torch.load(mPath))

final_conv = 'Mixed_7c'


# fine tuning
if PRETRAINED:
    for param in net.parameters():
        param.requires_grad = False
    net.fc = torch.nn.Linear(2048, len(classes))

net.cuda()


# load checkpoint
if RESUME != 0:
    print("===> Resuming from checkpoint.")
    assert os.path.isfile(camOutputDir+'checkpoint/'+ str(RESUME) + '.pt'), 'Error: no checkpoint found!'
    net.load_state_dict(torch.load(camOutputDir+'checkpoint/' + str(RESUME) + '.pt'))


# retrain
criterion = torch.nn.CrossEntropyLoss()

if PRETRAINED:
    optimizer = torch.optim.SGD(net.fc.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
else:
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

for epoch in range (1, EPOCH + 1):
    print('at epoch {}'.format(EPOCH))
    retrain(trainloader, net, USE_CUDA, epoch, criterion, optimizer)
    retest(testloader, net, USE_CUDA, criterion, epoch, RESUME,camOutputDir)


# hook the feature extractor
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(final_conv).register_forward_hook(hook_feature)


# CAM
base_transform1 = transforms.Compose([
    transforms.ToTensor()
])
test_data = ImageFolderWithPaths(root=testingPath, transform=base_transform1)
testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

if CAM:
    for i,j,k in testloader:
        print(len(k))
        images=k;
        for root in images:
            #root = 'sample1.jpg'
            img = Image.open(root)
            filename,classId=root.split('/')[-1],root.split('/')[-2]
            outputPath=camOutputDir+'camImages/'+classId+'_'+filename;

            get_cam(net, features_blobs, img, classes, root,outputPath)
