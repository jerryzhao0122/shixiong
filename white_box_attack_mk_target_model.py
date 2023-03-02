'''
time:2022/4/25
author:Guo Zhenyuan
'''

import os
import copy
import time
import numpy as np
import torch 
import warnings

from tensorboardX import SummaryWriter

from configs.fedl_params import fl_params
from configs.model_params import ml_params
from fedlearn.client import Attacker, Clients
from fedlearn.server import Server  
from model.cnn import CNNCifar, Mnist
from utils.data import get_dataset
from utils.sampling import iid,noniid_equal,noniid_unequal,noniid, root_data, whitebox
from torch.utils.data import DataLoader
from model.alexnet import AlexNet
from model.resnet import ResNet

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    fl_p = fl_params()

    # 获取数据
    dataset, test = get_dataset(fl_p)
    # print(len(dataset))

    #根据论文划分数据
    dataset.data = dataset.data[:30000]
    dataset.targets = dataset.targets[:30000]
    
    train_loaders = whitebox(dataset,fl_p)
    test_loaders = DataLoader(test,batch_size=fl_p.client_bs,shuffle=True)

    start_model = start_model = CNNCifar(fl_p).cuda()

    # 初始化Server
    server = Server(start_model,test_loaders,fl_p)

    # 初始化Clients
    clients = Clients(train_loaders,fl_p)  

    for r in range(1,fl_p.round+1):
        print('Round: {} Start'.format(r))
        #进行联邦学习
        broadcast_weight = server.broadcast()
        update_weights = clients.update(start_model,copy.deepcopy(broadcast_weight))
        torch.cuda.empty_cache()
        server.aggregation(copy.deepcopy(update_weights))

        #学习的信息展示
        server.printinfo()

        #写入tensorboard
        server.writerLogs()
        print('Round: {} End'.format(r))

        # if r in [100,150,200,250,300]:
        #     print('Start Save model')
        #     server.saveMode()  
    

   
