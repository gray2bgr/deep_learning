#-*- encoding:utf-8 -*-
import torch
import torchvision
from torchvision import transforms,datasets,models
import os
import matplotlib.pyplot as plt
import  numpy as np
import time
import copy
import json
from torch import nn,optim
from train_model import train_model
 
#transforms for trainning
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # Cutout(n_holes=1, length=8)
    ]),
}
data_dir = u'../data/train'  # train data
class2idx = u'../data/class2idx.json'
BATCH_SIZE = 48
train_datas = datasets.ImageFolder(root=data_dir,transform=data_transforms['train'])
 
class_names = train_datas.classes
num_classes  = len(train_datas.class_to_idx)
num_images = len(train_datas)
#overview datasets
print(class_names)
print('num of images: {}'.format(str(num_images)))
print('num of classes: {}'.format(str(num_classes)))
 
#save class to idx
if not os.path.exists(class2idx):
    with open(class2idx,'w') as f_json:
        f_json.write(json.dumps(train_datas.class_to_idx))
 
 
data_loaders1 = torch.utils.data.DataLoader(train_datas,batch_size=BATCH_SIZE,shuffle=True)
data_loaders2 = torch.utils.data.DataLoader(train_datas,batch_size=BATCH_SIZE,shuffle=True)
 
#model for out datasets
# net = models.resnet152(pretrained=True,)
# num_in_features = net.fc.in_features
# net.fc = nn.Linear(num_in_features,num_classes)
net = models.densenet201(pretrained=True)  #using densenet169 for training our datasets
num_in_features = net.classifier.in_features     #fit output for our datasets
net.classifier = nn.Linear(num_in_features, num_classes)
 
criterion = nn.CrossEntropyLoss()
 
optimizer = optim.Adam(net.parameters(),lr=1e-5)
 
 
train_model(model=net,
            criterion=criterion,
            optimizer=optimizer,
            data_loaders1=data_loaders1,
            data_loaders2=data_loaders2,
            num_images=num_images,
            cuda_device=True,
            finetune='best',
            num_epochs=100)