import os
import glob
import time
import random
from turtle import forward
from cv2 import dnn_DetectionModel
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F  
import torchvision
import matplotlib.pyplot as plt
import shutil
import pickle
from PIL import Image # 8.0.1
import argparse
import subprocess
import wandb

"""
To limit the usage of RAM and computation

Justification:
The significance of gradient (as well as activation) computations 
for a membership inference attack varies over the layers of a deep neural 
network. The first layers tend to contain less information about the specific 
data points in the training set, compared to non-member data record from 
the same underlying distribution.

Nasr, Milad, Reza Shokri, and Amir Houmansadr. “Comprehensive Privacy Analysis of 
Deep Learning: Passive and Active White-Box Inference Attacks against Centralized 
and Federated Learning.” 2019 IEEE Symposium on Security and Privacy (SP), May 
2019, 739–53. https://doi.org/10.1109/SP.2019.00065.
"""
selected_conv_layer_names = ['conv1','conv2','fc2','fc3']

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=50000, 
        help="""Only the first N samples of defender and reserve data will be used, 
        this means 2 * N samples in total.""")
    parser.add_argument("--input_feature_path", type=str, 
        default="/mnt/models/feature", 
        help="""where to load the input feature of the white-box attacker neural network.""")
    parser.add_argument("--save_attacker_model_path", type=str, 
        default="attacker_NN_model_checkpoints", 
        help="""where to save the checkpoints of the white-box attacker neural network.""")
    parser.add_argument('--random_seed', type=int, help='random seed', default=68)
    parser.add_argument('--lr', type=float, help='learning rate of the optimizer', default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--zip", action="store_true", default=False)
    args = parser.parse_args()
    return args

def load_input_features(args):
    """
    L_all: 
        list of python scalars

    hidden_all: 
        list of dict, each dict: str ==> 3D float32 np.array

    gradients_all: 
        list of dict, each dict: str ==> 4D float32 np.array

    y_all: 
        numpy.ndarray (2 * N, 10), float32 (one-hot encoding)

    yhat_all: 
        numpy.ndarray, shape = (2 * N, 10), float32

    input_features:
        list of InputFeature objects, 2 * N in total
    """
    with open(os.path.join(args.input_feature_path, "L_all.pkl"), "rb") as f:
        L_all = pickle.load(f) 

    with open(os.path.join(args.input_feature_path, "hidden_all.pkl"), "rb") as f:
        hidden_all = pickle.load(f) 

    with open(os.path.join(args.input_feature_path, "gradients_all.pkl"), "rb") as f:
       gradients_all = pickle.load(f) 

    with open(os.path.join(args.input_feature_path, "y_all.npy"), "rb") as f:
        y_all = np.load(f)

    with open(os.path.join(args.input_feature_path, "yhat_all.npy"), "rb") as f:
        yhat_all = np.load(f)

    input_features = group_input_features(args, L_all, hidden_all, y_all, yhat_all,gradients_all)

    return input_features

def group_input_features(args, L_all, hidden_all, y_all, yhat_all, gradients_all):
    """
    concatenate L_all, y_all, yhat_all to one flatten vector
    to be fed into a fully-connected feature extractor
    """
    # assert len(L_all) == args.N * 2
    # assert len(hidden_all) == args.N * 2
    # #assert len(gradients_all) == args.N * 2
    # assert len(y_all) == args.N * 2
    # assert len(yhat_all) == args.N * 2

    input_features = []
    print(len(L_all))

    for i in range(args.N):
        L = L_all[i]
        hidden = hidden_all[i]
        grad = gradients_all[i]
        y = y_all[i]
        yhat = yhat_all[i]

        input_feature = InputFeature(L, hidden, y, yhat,grad)
        input_features.append(input_feature)
    return input_features

class InputFeature:
    """
    L: python float32 scalar
    hidden: dict from str to 3D float32 np.array
    grad: dict from str to 4D float32 np.array
    y: np.ndarray of size 10, float32 (one-hot encoding)
    yhat: np.ndarray of size 10, float32 
    """
    def __init__(self, L, hidden, y, yhat,grad):
        self.flatten_vector = [L]
        self.flatten_vector.extend(y.tolist())
        self.flatten_vector.extend(yhat.tolist())
        self.flatten_vector = np.array(self.flatten_vector).astype(np.float32)

        self.keys = []
        for k in grad.keys():
           self.keys.append(k)

        self.grad = grad
        self.hidden = hidden

class FullyConnectedFeatureExtractor(nn.Module):
    def __init__(self, dim_in, dim_out=64):
        super().__init__() 
        self.fc1 = nn.Linear(dim_in, 128)
        self.fc2 = nn.Linear(128, dim_out)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        return x

class FullyConnectedEncoder(nn.Module):
    def __init__(self, dim_in, dim_out=2):
        super().__init__() 
        self.fc1 = nn.Linear(dim_in, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,dim_out)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.relu(self.fc3(self.dropout(x)))
        x = F.relu(self.fc4(self.dropout(x)))
        # x = F.softmax(x)
        return x

class Convolutional2DFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels,w,h):
        super().__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.fc = FullyConnectedFeatureExtractor(out_channels*(w-3+1)*(h-3+1),64)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv(self.dropout(x)))
        x = torch.flatten(x)
        x = F.relu(self.fc(self.dropout(x)))
        return x

class Convolutional3DFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels,d,w,h):
        super().__init__() 
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3)
        self.fc = FullyConnectedFeatureExtractor(out_channels*(d-3+1)*(w-3+1)*(h-3+1),64)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv(self.dropout(x)))
        x = torch.flatten(x)
        x = F.relu(self.fc(self.dropout(x)))
        return x

class WhiteBoxAttackerNeuralNetwork(nn.Module):
    def __init__(self, selected_conv_layer_names, hidden_layer_features_shape, param_gradients_shape, device):
        super().__init__() 
        self.device = device
        self.selected_conv_layer_names = selected_conv_layer_names

        self.grad_feature_extractors = {}
        for name in self.selected_conv_layer_names:
            if 'conv' in name:
                channels,d,h,w = param_gradients_shape[name]
                self.grad_feature_extractors[name] = Convolutional3DFeatureExtractor(channels, 2,d,h,w)
                self.grad_feature_extractors[name].to(self.device)
            else:
                channels=1
                w,h = param_gradients_shape[name]
                self.grad_feature_extractors[name] = Convolutional2DFeatureExtractor(1,2,w,h)
                self.grad_feature_extractors[name].to(self.device)

        
        self.hidden_feature_extractors = {}
        for name in self.selected_conv_layer_names:
            if 'conv' in name:
                channels,w,h = hidden_layer_features_shape[name]    
                dim_in = channels*w*h
                # print(dim_in)
                self.hidden_feature_extractors[name] = FullyConnectedFeatureExtractor(dim_in,64)
                self.hidden_feature_extractors[name].to(self.device)
            else:
                
                dim_in = hidden_layer_features_shape[name][0]
                # print(dim_in)
                self.hidden_feature_extractors[name] = FullyConnectedFeatureExtractor(dim_in,64)
                self.hidden_feature_extractors[name].to(self.device)

        self.fc_feature_extractor = FullyConnectedFeatureExtractor(21, 64).to(self.device)

        encoder_input_dim = 64 + 64*4 + 64*4

        self.encoder = FullyConnectedEncoder(encoder_input_dim,2).to(self.device)

    def forward(self,input_feature):
        fc_feature = torch.from_numpy(input_feature.flatten_vector).unsqueeze(0).to(self.device)
        x_list = [self.fc_feature_extractor(fc_feature).view(-1)]

        for name in self.selected_conv_layer_names:
            tmp = torch.from_numpy(input_feature.hidden[name]).view(-1).to(self.device)
            # print(tmp.shape)
            x_list.append(self.hidden_feature_extractors[name](tmp).view(-1))
            if 'fc' in name:
                x_list.append(self.grad_feature_extractors[name](torch.from_numpy(input_feature.grad[name]).unsqueeze(0).unsqueeze(0).to(self.device)))
            else:
                x_list.append(self.grad_feature_extractors[name](torch.from_numpy(input_feature.grad[name]).unsqueeze(0).to(self.device)))
        x = torch.cat(x_list,dim=0).view(-1)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        return x

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self,features,labels):
        super().__init__()
        self.features = features
        self.label = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self,idx):
        return self.features[idx],self.label[idx]

class FeatureDataloader:
    def __init__(self,dataset,shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle

        self.current_iteration_indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(self.current_iteration_indices)
        self.current_idx = 0
    
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.dataset):
            self.current_idx = 0
            if self.shuffle:
                np.random.shuffle(self.current_iteration_indices)
            raise StopIteration
        X,y = self.dataset[self.current_iteration_indices[self.current_idx]]
        self.current_idx+=1
        return X,y

def get_data_loaders(args,input_features):

    # N = (args.N - 20000) // 4

    defender_train = input_features[:15000]
    defender_test = input_features[15000:20000]
    reserve_train = input_features[-15000:]
    reserve_test = input_features[-20000:-15000]

    train_features = defender_train + reserve_train
    train_labels = np.ones(30000,dtype=np.int64)
    train_labels[-15000:]=0

    test_features = defender_test + reserve_test
    test_labels = np.ones(30000,dtype=np.int64)
    test_labels[-15000:]=0

    train_dataset = FeatureDataset(train_features, train_labels)
    test_dataset = FeatureDataset(test_features, test_labels)

    train_dataloader = FeatureDataloader(train_dataset, shuffle=True)
    test_dataloader = FeatureDataloader(test_dataset, shuffle=False)
    return train_dataloader, test_dataloader

def train(train_dataloader, model, device, optim, epoch, args):
    t0 = time.time()

    train_loss = 0
    correct_cnt = 0
    total_cnt = 0

    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for batch_idx, (X, y) in enumerate(train_dataloader):
        y = torch.from_numpy(np.array([y])).to(device)

        optim.zero_grad()

        logits = model(X).unsqueeze(0)
        loss = criterion(logits, y)
        # print(logits)
        loss.backward()
        optim.step()

        train_loss += loss.item()
        correct_cnt += (logits.argmax(dim=1) == y).sum().item() 
        total_cnt += 1
        # print(logits)

    train_loss /= len(train_dataloader)
    train_acc = (correct_cnt / total_cnt) * 100
    
    t1 = time.time() - t0
    print("Epoch {} | Train loss {:.2f} | Train acc {:.2f} | Time {:.1f} seconds.".format(
            epoch+1, train_loss, train_acc, t1))
    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc, "train_epoch_time": t1})


def test(test_dataloader, model, device, epoch, args, test_logits):
    t0 = time.time()
    
    test_loss = 0
    correct_cnt = 0
    total_cnt = 0

    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_dataloader):
            y = torch.from_numpy(np.array([y])).to(device)
            
            logits = model(X).unsqueeze(0)

            test_logits.extend(logits.detach().cpu().numpy())
            
            loss = criterion(logits, y)

            test_loss += loss.item()
            correct_cnt += (logits.argmax(dim=1) == y).sum().item() 
            total_cnt += 1

    
    test_loss /= len(test_dataloader)
    test_acc = (correct_cnt / total_cnt) * 100

    
    t1 = time.time() - t0
    print("Epoch {} | Test loss {:.2f} | Test acc {:.2f} | Time {:.1f} seconds.".format(
            epoch+1, test_loss, test_acc, t1))
    wandb.log({"epoch": epoch+1, "test_loss": test_loss, "test_acc": test_acc, "test_epoch_time": t1})
    wandb.run.summary["final_test_loss"] = test_loss
    wandb.run.summary["final_test_acc"] = test_acc

def get_torch_gpu_environment():
    env_info = dict()
    env_info["PyTorch_version"] = torch.__version__

    if torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["cuDNN_version"] = torch.backends.cudnn.version()
        env_info["nb_available_GPUs"] = torch.cuda.device_count()
        env_info["current_GPU_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        env_info["nb_available_GPUs"] = 0
    return env_info

def count_trainable_parameters(model):
    return sum([x.numel() for x in model.parameters() if x.requires_grad])

if __name__ == "__main__":

    t0 = time.time()

    args = parse_arguments()

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Using CPU for PyTorch")
    else:
        device = torch.device("cuda")
        print("Using GPU for PyTorch") 

    # wandb
    project_name = "white_box_membership_attacker_with_NN_test"
    group_name = "{}".format(args.lr)
    wandb_dir = "/mnt/wandb_logs"
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    wandb.init(config=args, project=project_name, group=group_name, dir=wandb_dir)
    
    env_info = get_torch_gpu_environment()
    for k, v in env_info.items():
        wandb.run.summary[k] = v
    wandb_run_name = wandb.run.name   

    # load input data
    input_features = load_input_features(args)

    # print(input_features[7000].grad['conv1'])
    # print(input_features[5000].grad['conv1'])

    print("input_features loaded.")

    # dataloaders
    train_dataloader, test_dataloader = get_data_loaders(args, input_features)

    print("Dataloaders ready.")

    # 计算获得隐藏层和梯度层特征维度
    grad_shape = {}
    hidden_shape = {}
    for name in selected_conv_layer_names:
        grad_shape[name] = input_features[0].grad[name].shape
        hidden_shape[name] = input_features[0].hidden[name].shape
    print(grad_shape)
    print(hidden_shape)


    # load model
    model = WhiteBoxAttackerNeuralNetwork(selected_conv_layer_names,hidden_shape,grad_shape,device)

    print("Model ready.")


    wandb.run.summary["trainable_parameters_count"] = count_trainable_parameters(model)



    # optimizer
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    print("Optimizer ready.")

    white_box_attack_NN_results_dir = "/mnt/white_box_attack_NN_results"
    if not os.path.exists(white_box_attack_NN_results_dir):
        os.makedirs(white_box_attack_NN_results_dir)
    
    test_logits = []

    # train
    print("Training loop starts...")
    for epoch in range(args.epochs):
        train(train_dataloader, model, device, optim, epoch, args)
        test(test_dataloader, model, device, epoch, args, test_logits)

        wandb.log({"epoch": epoch+1, "lr": optim.param_groups[0]['lr']})

    print("Training loop ends.")

    torch.save(model.state_dict(), 
        os.path.join(white_box_attack_NN_results_dir, "lanet_{}.pth".format(wandb_run_name)))

    # save test_logits with size N (3000)
    ## the true label of the first half is 1 (defender), the true label of the second half is 0 (reserve).
    with open(os.path.join(white_box_attack_NN_results_dir, "test_logits.pkl"), "wb") as f:
        pickle.dump(test_logits, f)


    if args.zip:
        print("Starts zipping the directory {}".format(white_box_attack_NN_results_dir))
        cmd = "zip -r {}.zip {}".format(white_box_attack_NN_results_dir, white_box_attack_NN_results_dir)
        subprocess.call(cmd.split())

    print("Done in {:.1f} s.".format(time.time() - t0))

    # print(len(input_features))
    # print(input_features[0].grad['conv2'].shape)
    # print(input_features[0].hidden['conv2'].shape)
    # print(input_features[0].flatten_vector.shape)

    # grad = input_features[0].grad
    # in_dim,d_dim,h_dim,w_dim = grad['conv2'].shape
     
    # conv = Convolutional3DFeatureExtractor(in_dim,1000,d_dim,h_dim,w_dim)
    # kk = conv(torch.from_numpy(grad['conv2']).unsqueeze(dim=0))
    # print(kk.shape)

    # fc = FullyConnectedFeatureExtractor(21, 2)
    # tt = fc(torch.from_numpy(input_features[0].flatten_vector))
    # print(tt)

    # hd = FullyConnectedFeatureExtractor(16*10*10,64)
    # ll = hd(torch.from_numpy(input_features[0].hidden['conv2']).view(-1))
    # print(ll)

    # # 计算获得隐藏层和梯度层特征维度
    # grad_shape = {}
    # hidden_shape = {}
    # for name in selected_conv_layer_names:
    #     grad_shape[name] = input_features[0].grad[name].shape
    #     hidden_shape[name] = input_features[0].hidden[name].shape
    # print(grad_shape)
    # print(hidden_shape)
    
    # wb = WhiteBoxAttackerNeuralNetwork(selected_conv_layer_names,hidden_shape,grad_shape,device)
    # jj = wb(input_features[0])
    # print(jj)