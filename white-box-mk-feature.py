from logging import root
from cv2 import transform
import torch 
import torch.nn as nn
import torch.nn.functional as F  
import torchvision 
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader  
from torchvision import datasets, transforms
import pickle
import numpy as np
import argparse
import os

from tqdm import tqdm

from configs.fedl_params import fl_params
from model.cnn import CNNCifar

# selected_conv_layer_names = ['conv1','conv2','fc1','fc2','fc3']
selected_conv_layer_names = ['layer1.0','layer2.0','layer3.0','layer4.0','layer5.0',
                            'fc1.1','fc2.1','fc3.1']

#定义Alexnet网路结构
class AlexNet(torch.nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(AlexNet, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(3, 96, kernel_size=5, stride=2, padding=2),
                                          # raw kernel_size=11, stride=4, padding=2. For use img size 224 * 224.
                                          torch.nn.BatchNorm2d(96),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                                          torch.nn.BatchNorm2d(256),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(384),
                                          torch.nn.ReLU(inplace=True))
        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(384),
                                          torch.nn.ReLU(inplace=True))
        self.layer5 = torch.nn.Sequential(torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(384),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = torch.nn.Sequential(torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(384 * 2 * 2, 4096),
                                       torch.nn.ReLU(inplace=True))
        self.fc2 = torch.nn.Sequential(torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(inplace=True))
        self.fc3 = torch.nn.Sequential(torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

def load_trained_model(model_path, device):
    arg = fl_params()
    model = AlexNet()
    model.to(device)
    model.load_state_dict(torch.load(model_path,map_location=device))
    return model

def convert_to_one_hot_encoding(y):
    one_hot = np.zeros(10, dtype=np.int64)
    one_hot[y] = 1
    return one_hot

def load_trained_data():
    transform = transforms.Compose([
                                    transforms.ToTensor()
                                    ])

    train_dataset = torchvision.datasets.CIFAR10(
                root="/home/featurize/data",
                train=True,
                download=True,
                transform=transform
            )
    
    train_dl = DataLoader(train_dataset,batch_size=128,num_workers=6)

    return train_dl,train_dataset

def save_pickle_object(results_dir, file_name, obj):
    file_path = os.path.join(results_dir, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
    print("{} saved.".format(file_path))

def save_np_array(results_dir, file_name, arr):
    file_path = os.path.join(results_dir, file_name)
    with open(file_path, "wb") as f:
        np.save(f, arr)
    print("{} saved.".format(file_path)) 

def y_all(train_ds,results_dir):
    y_all = [convert_to_one_hot_encoding(i) for i in train_ds.targets]

    y_all = np.array(y_all)

    
    save_np_array(results_dir,y_all,y_all.astype('float32'))

def y_hat(model,train_dl,device):

    model.eval()
    res = []
    for batch_idx, (images, labels) in enumerate(train_dl):
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        res.append(output)
    
    y_hat = torch.cat(res)

    y_hat = np.array(y_hat)

    return y_hat
    
def 
