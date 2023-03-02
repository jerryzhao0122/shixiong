'''
time:2022/7/3
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
from fedlearn.client import Attacker, Clients
from fedlearn.server import Server  
from model.cnn import CNNCifar, Mnist
from model.resnet import ResNet
from utils.data import get_dataset
from utils.sampling import iid,noniid
from torch.utils.data import DataLoader

from model.alexnet import AlexNet

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    fl_p = fl_params()

    # 获取数据
    train_dataset,test_dataset = get_dataset(fl_p)
    print('数据获取完毕')

    if fl_p.data_distributed == 'iid':
        train_loaders = iid(train_dataset,fl_p)
    elif fl_p.data_distributed == 'noniid':
        train_loaders = noniid(train_dataset,fl_p)
    else:
        print('请输入正确的数据分布类型')

    print('dataloader划分完毕')
    test_loader= DataLoader(test_dataset,batch_size=fl_p.client_bs,shuffle=True)

    print('开始加载模型：',fl_p.model)
    if fl_p.model == 'resnet':
        start_model = ResNet().cuda()
    elif fl_p.model == 'lanet':
        if fl_p.dataset == 'cifar10':
            start_model = CNNCifar(fl_p).cuda()
        else:
            start_model = Mnist(fl_p).cuda()
    elif fl_p.model == 'alexnet':
        start_model = AlexNet(num_classes=10,track=False).cuda()
    else:
        pass
    print('加载模型完毕')

    # 原始代码
    # if fl_p.dataset == 'cifar10':
    #     print('cifar model')
    #     start_model = CNNCifar(fl_p).cuda()
    # else:
    #     print('mnist model')
    #     start_model = Mnist(fl_p).cuda()
    # print('模型加载完毕')

    server = Server(start_model,test_loader,fl_p)
    clients = Clients(train_loaders,fl_p)

    if fl_p.attack == True:
        # 初始化攻击者cliet
        for i in range(fl_p.attack_num):
            attacker_data_loader = clients.all_clients[i].dataloader
            attacker = Attacker(i,attacker_data_loader,1,fl_p)
            clients.all_clients[i] = attacker
    else:
        print('错误，没有攻击者')
    
    for i in range(1,fl_p.round+1):
        print('Round: {} Start'.format(i))

        # 进行联邦学习
        broadcast_weight = server.broadcast()
        update_weights = clients.update(broadcast_weight)
        server.aggregation(update_weights)
        
        # 当前结果信息展示
        server.printinfo()

        # 写入tensorboardd
        server.writerLogs()
        print('Round: {} End'.format(i))
    
    if fl_p.save_mipc == True:
        print('Save mipc all results')
        name = '/home/featurize/result/verify_ad/' + fl_p.name + '.npy'
        np.save(name, server.mipc_results)
    print('okk--train---end')



        

