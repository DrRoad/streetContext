#this code was used to generate the tsne visualization plot
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
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import umap
import scipy
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, AnchoredOffsetbox
import torchvision


# %%
class vizHelper:

    def getImage(self,path, zoom=0.15):
        return OffsetImage(plt.imread(path), zoom=zoom)

    def getClosest(self,x,listOfPoints):
        i=-1;
        taken={}
        mindist=math.inf;
        for _ in range(len(listOfPoints)):
            dist=scipy.spatial.distance.euclidean(list(x), listOfPoints[_,:])
            if dist<mindist:
                mindist=dist;
                i=_;
        return mindist,listOfPoints[i,:],i

    def mapEmbedToGrid(self,X_embedded, paths, factor,threshold=1):
        new_X_em = X_embedded.copy();
        x,y = [i for i in X_embedded[:, 0]],[i for i in X_embedded[:, 1]]

        #make a grid
        min_x, max_x, min_y, max_y = min(x), max(x), min(y), max(y)
        a, b = np.meshgrid(np.linspace(min_x, max_x, factor[0]), np.linspace(min_y, max_y, factor[1]))
        a,b = a.flatten(),b.flatten()
        grid = np.array([a, b]).transpose()
        gridAssign=defaultdict(list)
        seen=[]
        min_idx=-1;
        for _ in range(len(grid)):
            dist,point,pointidx = self.getClosest(grid[_, :], new_X_em)
            if dist<threshold and pointidx not in seen:
                gridAssign[pointidx]=_
                seen.append(pointidx)
                #new_X_em = np.delete(new_X_em, pointidx, 0);

        return gridAssign,grid


    def mapping(self,gridAssign,grid,paths,classes):
        new_paths,new_classes,new_embed=[],[],[]
        for i in gridAssign:
            #print(i,len(paths),len(new_paths))
            new_paths.append(paths[i])
            new_classes.append(classes[i])
            new_embed.append(grid[gridAssign[i],:])
        return new_paths,new_classes,np.array(new_embed)

    def loadModel(self,mPath):
        # read the model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = alexnet()#resnet_34()#torchvision.models.alexnet(pretrained=True)# resnet_18()
        model.load_state_dict(torch.load(mPath))
        model.eval()
        model.to(device)
        #
        before_last_model = torch.nn.Sequential(*list(model.children())[:-1])
        last_layer = list(model.children())[-1]
        return model,last_layer,before_last_model,device

    def getFeatures(self,mPath,dPath,threshold=0,samplesize=300):
        #
        home = expanduser("~")
        modelPath = home + mPath
        #get a model
        model,last_layer,before_last_model,device=self.loadModel(modelPath)
        #get data
        data_dir = home + dPath
        test_dataset = datasets.ImageFolder(root=data_dir)
        classeslabels = list(test_dataset.class_to_idx)

        #
        base_transform = transforms.Compose([transforms.ToTensor()])
        dataset = ImageFolderWithPaths(root=data_dir, transform=base_transform)  # our custom dataset
        dataloader = torch.utils.data.DataLoader(dataset)

        # iterate over data
        sampleCnt,count,classes,paths = [samplesize] * 10,0,[],[];
        features_extracted = np.zeros([samplesize * 10, 256*5*11])
        #resnet features_extracted = np.zeros([samplesize*10, 512])
        for input, label, path in dataloader:

            input = input.to(device)
            modelOutput = model(input)
            modelOutput1 = model(input)
            prediction = torch.argmax(modelOutput1, 1)
            label = label.to(device)
            if (prediction == label and sampleCnt[label] > 0) and np.random.random()>threshold:
                sampleCnt[label] -= 1;
                # use the above variables freely
                input = input.to(device)
                modelOutput = before_last_model(input)
                features_extracted[count, :] = modelOutput.cpu().data.numpy().reshape([1, 256*5*11])
                #resnet features_extracted[count, :] = modelOutput.cpu().data.numpy().reshape([1,512])

                classes.append(label.cpu().data.numpy()[0])
                paths.append( path[0])
                count += 1;
        features_extracted = features_extracted[:count, :];
        return features_extracted,paths,classes,classeslabels




#%%
mPath='/streetContexts/data/streetImages/Boston/modelRelated/models/AlexNet/AlexNet_model.500'
dPath='/streetContexts/data/streetImages/Boston/modelRelated/data/resnet/train'
helper=vizHelper()
features_extracted,paths,classes,classesText=helper.getFeatures(mPath,dPath,threshold=0,samplesize=250)

# %% changing some of the labels to make it a unsupervised UMAP
labels=[i for i in classes]
for i in range(len(classes)):
    if np.random.random()>0:
        labels[i]=-1;

# %% UMAP
X_embedded = umap.UMAP(n_components=2,n_neighbors=250,min_dist=0.1,metric='euclidean')\
             .fit_transform(np.asarray(features_extracted.data))#,y=labels)
# %% TSNE
X_embedded = TSNE(n_components=2,n_iter=100000,perplexity=100,early_exaggeration=1,metric='euclidean' ,verbose=2)\
                  .fit_transform(np.asarray(features_extracted.data))
# %% plot colored dots
plt.figure()
colors = cm.rainbow(np.linspace(0, 1, 10))
fig, ax = plt.subplots()
scatterColors=[colors[i] for i in classes]
for i in range(max(classes)):
    idx=[j for j in range(len(classes)) if i==classes[j]];
    ax.scatter(X_embedded[idx,0],X_embedded[idx,1],c=[colors[i]],label=classesText[i])

plt.show()
# %% 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = cm.rainbow(np.linspace(0, 1, 10))
scatterColors=[colors[i] for i in classes]
ax.scatter(X_embedded[:,0],X_embedded[:,1],X_embedded[:,2],c=classes,cmap='Spectral')
plt.show()

# %% plot the images
counts=10*[400]
paths=[i for i in paths]
gridAssign,grid=helper.mapEmbedToGrid(X_embedded=X_embedded,paths=paths,factor=[17,33],threshold=2)
#
helper=vizHelper()
paths_Mapped,classes_Mapped,X_embeddedMapped=helper.mapping(gridAssign,grid,paths,classes)
# 
fig, ax = plt.subplots(figsize=[16.5,17])
#ax.scatter(X_embeddedMapped[:,0],X_embeddedMapped[:,1], facecolors='none', edgecolors='none')
for i in range(max(classes)+1):
    idx=[j for j in range(len(classes)) if i==classes[j]];
    ax.scatter(X_embedded[idx,0],X_embedded[idx,1],c=[colors[i]],label=classesText[i])
    ax.scatter(X_embedded[idx,0],X_embedded[idx,1],c='white')
ax.legend(prop={'size':15})


artists = []
for pts, path, class_ in zip(X_embeddedMapped,paths_Mapped,classes_Mapped):
    x0,y0=pts
    if x0<np.inf and counts[class_]>0 and np.random.random()>0.00:
        ab = AnnotationBbox(helper.getImage(path,zoom=0.30), (x0, y0),box_alignment=(0,0), pad=0.1, bboxprops =dict(edgecolor=colors[class_]),frameon=True)
        artists.append(ax.add_artist(ab))
        counts[class_] -=1;

fig.show()
fig.savefig('/home/fahad/streetContexts/paper/plots/tsne1.pdf')


# %% plot the images
counts=10*[3]
paths=[i for i in paths]
gridAssign,grid=helper.mapEmbedToGrid(X_embedded=X_embedded,paths=paths,factor=[20,40],threshold=2)
#
helper=vizHelper()
paths_Mapped,classes_Mapped,X_embeddedMapped=helper.mapping(gridAssign,grid,paths,classes)
#
fig, ax = plt.subplots(figsize=[10,10])
#ax.scatter(X_embeddedMapped[:,0],X_embeddedMapped[:,1], facecolors='none', edgecolors='none')
scatterColors=[colors[i] for i in classes]
ax.scatter(X_embedded[:,0],X_embedded[:,1],c=scatterColors,leg)

artists = []
for pts, path, class_ in zip(X_embedded,paths,classes):
    x0,y0=pts
    if x0<np.inf and counts[class_]>0:

        ab = AnnotationBbox(helper.getImage(path,zoom=0.23), (x0, y0),box_alignment=(0,0), pad=0.0, bboxprops =dict(edgecolor=colors[class_]),frameon=True)
        artists.append(ax.add_artist(ab))
        counts[class_] -=1;
fig.show()

