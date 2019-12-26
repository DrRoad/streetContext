import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform
import PIL
from models.AlexNet import *
from models.ResNet import *
import os


def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()


dataPath='/home/fahad/data/streetContext/streetImages/Boston/modelRelated/data/resnetPOI/train'
CAMPath='/home/fahad/data/streetContext/streetImages/Boston/modelRelated/cam/camImages'
ClassesFolders=[i for i in os.listdir(dataPath)]

# Imagenet mean/std

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

# Preprocessing - scale to 224x224 for model, convert to tensor,
# and normalize to -1..1 with mean/std for ImageNet

base_transform = transforms.Compose([
    transforms.ToTensor()
    ])

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

display_transform = transforms.Compose([
   transforms.Resize((224,224))])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#loading the model
startFromPath='/home/fahad/data/streetContext/streetImages/Boston/modelRelated/models/resnet18/BOS_POI_resnet18.880'
model = resnet_18()
model.load_state_dict(torch.load(startFromPath))
model.cuda()
model.eval()
model.to(device)

# Then looping over images
for folder in ClassesFolders:
    for imageName in os.listdir(dataPath + folder):
        imagePath = dataPath +'/'+ folder + '/' + imageName;
        targetPath = CAMPath +'/'+ folder + '/' + imageName;


        if not os.path.exists(targetPath):
            # change this to get a path for every image

            image = Image.open(imagePath)
            # imshow(image)

            tensor = base_transform(image)
            prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)
            prediction_var=prediction_var.to(device)
            final_layer = model._modules.get('layer4')

            activated_features = SaveFeatures(final_layer)

            prediction = model(prediction_var)
            pred_probabilities = F.softmax(prediction).data.squeeze()
            activated_features.remove()

            topk(pred_probabilities, 1)
            weight_softmax_params = list(model._modules.get('fc').parameters())
            weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
            class_idx = topk(pred_probabilities, 1)[1].int()
            overlay = getCAM(activated_features.features, weight_softmax, class_idx)
            fig1 = plt.gcf()
            plt.imshow(image)
            plt.imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.5, cmap='jet');
            fig1.savefig(str(CAMPath + '/' + folder + '/' + imageName))
            print('{} wrote the file {}'.format(os.path.exists(targetPath),str(CAMPath + '/' + folder + '/' + imageName)))
        else:
            print('Exists {}'.format(targetPath))