from cv2 import transform
import torch 
import torch.nn as nn
import torch.nn.functional as F  
import torchvision 
from torchvision.transforms import ToTensor
import pickle
import numpy as np
import argparse
import os

from tqdm import tqdm

from configs.fedl_params import fl_params
from model.cnn import CNNCifar

# selected_conv_layer_names = ['conv1','conv2','fc1','fc2','fc3']
selected_conv_layer_names = ['layer1.0','layer2.0','layer3.0','layer4.0','layer5.0']


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.relu(self.fc3(self.dropout(x)))
        return F.log_softmax(x, dim=1)

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

def get_transform():
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            lambda x: torch.unsqueeze(x, 0),
            lambda x: x.to(device)
        ]
    )
    return transform

def x_all_y_all(results_dir):
    all_dataset = torchvision.datasets.CIFAR10(
            root="/home/featurize/data",
            download=True
        )
    x_all = all_dataset.data
    y_all = [convert_to_one_hot_encoding(i) for i in all_dataset.targets]

    x_all = np.array(x_all)
    y_all = np.array(y_all)
    # save_np_array(results_dir, "y_all.npy", y_all.astype("float32"))
    save_np_array(results_dir, "y_all.npy", y_all.astype("float32"))
    return x_all,y_all

def evaluate_model(model, data, transform, args=None):
    """
    res: predicted probabilities
    """
    model.eval()
    res = []
    for i in range(data.shape[0]):
        img = transform(data[i])
        # if args.model_path in ["supervised_model_checkpoints/resnet50_fm_defender.pth", 
        # "supervised_model_checkpoints/resnet50_large_fm_defender.pth"]:
        #     res.append(F.softmax(model.predict(img).squeeze(0), dim=0).detach().to("cpu").numpy())
        # else:
        res.append(model(img).squeeze(0).detach().to("cpu").numpy())
        
    res = np.array(res)
    return res

def compute_yhat_all(model_path, device, x_all, transform, results_dir=None,arg=None):
    """
    yhat_all:
        numpy.ndarray, shape = (2 * N, 10), float32
    """
    model = load_trained_model(model_path, device)
    yhat_all = evaluate_model(model, x_all, transform)
    save_np_array(results_dir, "yhat_all.npy", yhat_all)
    return yhat_all

class HiddenLayerFeatures:
    """
    one object for each data point (x, y).
    
    If there are 2*N data points (defender+reserve), 
    then 2*N such objects need to be instantiated. 
    """
    def __init__(self, idx):
        self.idx = idx
        self.data = {}
        
    def get_hook(self, name):
        def hook(model, input, output):
            # .squeeze(dim=0) because of batch_size = 1
            # .cpu().numpy() because of "Out of CUDA memory" error
            if name in selected_conv_layer_names:
                self.data[name] = output.detach().squeeze(dim=0).cpu().numpy()
                # print(output.detach().squeeze(dim=0).cpu().numpy())
        return hook

def compute_the_other_attack_features_one_sample(model_path, device, transform, img, label, idx, iter_source=None):
    # img: torch.Tensor, torch.Size([1, 3, 224, 224]), torch.float32
    # label: torch.Tensor, shape = (1,), int64
    model = load_trained_model(model_path, device)

    params = model.parameters()

    optimizer = torch.optim.SGD(model.parameters(),lr = 0.02, momentum=0.5)    
    
    # set up forward_hooks
    hidden_layer_features = HiddenLayerFeatures(idx)
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            # print(name)
            layer.register_forward_hook(hidden_layer_features.get_hook(name))
        if isinstance(layer, torch.nn.modules.linear.Linear):
            # print(name)
            layer.register_forward_hook(hidden_layer_features.get_hook(name))

    model.train()

    gradients_layer = {}
    
    output = model(img)

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, label)

    L = loss.item()
    optimizer.zero_grad()
    loss.backward()

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            if name in selected_conv_layer_names:
                # layer.weight.grad: initially 4D tensor, torch.float32
                # print(name)
                gradients_layer[name] = layer.weight.grad.detach().cpu().numpy()
                # print(layer.weight.grad.detach().cpu().numpy())
        if isinstance(layer, torch.nn.modules.linear.Linear):
            if name in selected_conv_layer_names:
                # print(name)
                gradients_layer[name] = layer.weight.grad.detach().cpu().numpy()
                # print(layer.weight.grad.detach().cpu().numpy())
    # print(L)
    
    return L, hidden_layer_features.data, gradients_layer

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

def compute_the_other_attack_features(model_path, device, x_all, y_all, transform, results_dir=None):
    L_all = []
    hidden_all = []
    gradients_all = []
    
    for idx in tqdm(range(x_all.shape[0])):
        
        img = transform(x_all[idx]) 
        label = torch.from_numpy(np.expand_dims(np.argmax(y_all[idx]), axis=0)).to(device)
        L, hidden_layer_features_data, gradients_layer = compute_the_other_attack_features_one_sample(model_path, 
            device, transform, img, label, idx)
        
        L_all.append(L) # list of python scalars
        hidden_all.append(hidden_layer_features_data) # list of dict, each dict: str ==> 3D float32 np.array
        gradients_all.append(gradients_layer) # list of dict, each dict: str ==> 4D float32 np.array

        # print('hidden',hidden_all[-1])
        # print('grad',gradients_all[-1])

    save_pickle_object(results_dir, "L_all.pkl", L_all)
    save_pickle_object(results_dir, "hidden_all.pkl", hidden_all)
    save_pickle_object(results_dir, "gradients_all.pkl", gradients_all)

    return L_all, hidden_all, gradients_all

if __name__ == '__main__':

    model_path = '/home/featurize/result/models/white_box_target_test_alexnet300.pth.tar'
    results_dir = '/home/featurize/result/feature'

    x_all,y_all = x_all_y_all(results_dir)

    transform = get_transform()


    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Using CPU for PyTorch")
    else:
        device = torch.device("cuda")
        print("Using GPU for PyTorch")

    yhat_all = compute_yhat_all(model_path, device, x_all, transform, results_dir)

    L, hidden_layer_featuresdata, gradients_layer = compute_the_other_attack_features(model_path, 
        device, x_all, y_all, transform,results_dir)
    
    
    