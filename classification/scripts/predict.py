import torch
import torch.nn as nn
import os
import cv2
import json
from  tqdm import tqdm
from  PIL import Image
from torchvision import models,transforms
from torch.utils import data
 
 
class Dataset(data.Dataset):
    def __init__(self,image_path):
        self.image_path = image_path
        self.images_list = self.get_images(self.image_path)
        self.data_transforms ={
                        'val': transforms.Compose([
                            transforms.Scale(256),
                            transforms.RandomCrop(224),
                            # transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])}
    def __getitem__(self, index):
        image_name = self.images_list[index]
        input_image = Image.open(os.path.join(self.image_path,image_name))
        input_tensor = self.data_transforms['val'](input_image)
        # input_tensor = input_tensor.unsqueeze(0)
        return  input_tensor,image_name.replace('.png','')
 
    def __len__(self):
        return len(self.images_list)
 
    def get_images(self,image_path):
        return [i for  i in os.listdir(image_path)]
 
 
 
 
test_path = '../data/test'
 
test_dataset = Dataset(test_path)
 
#load class to dict
with open(u'../data/class2idx.json') as f_json:
    class_to_idx = json.load(f_json)
    print(type(class_to_idx))
    print(class_to_idx)
idx_to_class = {v: k for k, v in class_to_idx.items()}
 
num_classes = len(idx_to_class)
 
net = models.densenet201(pretrained=False)  # model for our datasets
num_in_features = net.classifier.in_features
net.classifier = nn.Linear(num_in_features, num_classes)  # fit densenet for our datasets
# net = models.resnet152(pretrained=False)
# num_in_features = net.fc.in_features
# net.fc = nn.Linear(num_in_features,num_classes)
 
USING_GPU = True
GPU_ID = 1
if not USING_GPU:
    # cup model
    net.load_state_dict(torch.load('../models/best_model',map_location='cpu'))
else:
 
    # gpu model
    net.load_state_dict(torch.load('../models/best_model'))
    net.cuda(GPU_ID)
net = net.eval()
BATCH_SIZE = 16
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)
with open('result.csv','w') as f_csv:
    f_csv.write('id,label\n')
    for inputs, names in tqdm(test_dataloader):
        if USING_GPU:
            inputs = inputs.cuda(GPU_ID)
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        for name,pred in zip(names,preds):
            f_csv.write('{},{}\n'.format(str(name),str(idx_to_class[int(pred.cpu().numpy())])))