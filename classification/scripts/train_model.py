import torch
import torchvision
from torchvision import transforms,datasets,models
import os
import  numpy as np
import time
import json
from torch import nn,optim
from torch.autograd import Variable
from tqdm import tqdm
 
 
def train_model(model, criterion, optimizer, data_loaders1,data_loaders2, num_images,cuda_device=False,finetune=None, num_epochs=25,CUDA_ID=0):
    '''
    :param model: model for training
    :param criterion: loss function
    :param optimizer:
    :param data_loaders:
    :param num_images: total numbers of training data
    :param cuda_device: True using cuda, False using cpu
    :param finetune:None for scratch, best finetune from best_model, last from last_model
    :param num_epochs: max iteration number
    :return:
    '''
    since = time.time()
    best_acc = 0.0
    begin_epoch = 0
    #save training log
    if not os.path.exists('../log'):
        os.makedirs('../log')
    now_time =  time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
    with open('../log/'+str(now_time)+'.txt','w') as f_log:
        if finetune:
            txt_path = '../models/{}_epoch.txt'.format(finetune)
            model_path = '../models/{}_model'.format(finetune)
            if os.path.exists(txt_path):
                with open(txt_path,'r') as f_epoch:
                    begin_epoch,best_acc = f_epoch.read().strip().split(',')
                    begin_epoch,best_acc = int(float(begin_epoch)),float(best_acc)
            if not os.path.exists(model_path):
                print("Cannot find {} !!!".format(model_path))
                print('Train from scratch...')
            else:
                model.load_state_dict(torch.load(model_path))
                print('Finetuning ....')
        else:
            print('Train from scratch...')
        for epoch in range(begin_epoch,num_epochs):
            model.train()
            if cuda_device:
                model= model.cuda(CUDA_ID)
 
            running_loss = 0.0
            running_acc = 0.0
 
            for (inputs, labels) in tqdm(data_loaders1):
                
                if cuda_device:
                    inputs = inputs.cuda(CUDA_ID)
                    labels = labels.cuda(CUDA_ID)
 
                inputs,labels = Variable(inputs),Variable(labels)
 
 
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                if cuda_device:
                    running_loss += float(loss.data.cpu().numpy())
                else:
                    running_loss += float(loss.data.numpy())
                running_acc += torch.sum(preds == labels)
                
            epoch_loss = running_loss / num_images
            if cuda_device:
                epoch_acc = running_acc.double().cpu().numpy() / num_images
            else:
                epoch_acc = running_acc.double().numpy() / num_images
                
 
            logs = 'Epoach at {}/{}, Train loss: {}, Acc: {}'.format(str(epoch),
                                                                    str(num_epochs),
                                                                    str(epoch_loss),
                                                                    str(epoch_acc))
            print(logs)
            f_log.write(logs)
 
            #save best and last model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(),'../models/best_model')
                with open('../models/best_epoch.txt','w') as f_best_epoch:
                    f_best_epoch.write(str(epoch)+','+str(best_acc))
            torch.save(model,'../models/last_model')
            with open('../models/last_epoch.txt', 'w') as f_best_epoch:
                f_best_epoch.write(str(epoch)+','+str(epoch_acc))
    now = time.time() - since
    print('Epoch {}/{}, time cost: {}s'.format(str(epoch),str(num_epochs),str(now)))