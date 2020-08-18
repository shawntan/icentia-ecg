import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import numpy as np
import sys,os
import pickle, gzip
import pandas as pd
import matplotlib,matplotlib.pyplot as plt
import sklearn, sklearn.model_selection, sklearn.neighbors
import sklearn.linear_model, sklearn.ensemble
import collections


################
## models mostly come from this code https://github.com/hsd1503/resnet1d


class MLP(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, out_channels, n_classes, seed=0):
        super(MLP, self).__init__()
        
        torch.manual_seed(seed)
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # (batch, channels, length)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, n_classes)
        )
        
    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x
    
class CNN(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, out_channels,  n_layers, n_classes, final_layer, kernel, stride, seed=0):
        super(CNN, self).__init__()
        
        torch.manual_seed(seed)
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.final_layer = final_layer
        self.kernel =kernel
        self.stride = stride
        self.printed = False
        
        # (batch, channels, length)
        layers_dict = collections.OrderedDict()
        
        layers_dict["conv0"] = nn.Conv1d(in_channels=self.in_channels, 
                                out_channels=self.out_channels, 
                                kernel_size=200, 
                                stride=self.stride)
        layers_dict["relu0"] = nn.ReLU()
        
        #last_size = 0
        for l in range(1,n_layers):
            layers_dict["conv{}".format(l)] = nn.Conv1d(in_channels=self.out_channels, 
                                    out_channels=self.out_channels, 
                                    kernel_size=kernel,
                                    stride=self.stride)
            layers_dict["relu{}".format(l)] = nn.ReLU()
            #last_size=(self.out_channels//(l+1))
            #print(last_size)
            
        self.layers = nn.Sequential(layers_dict)
        
        #print(self.layers)
        
        self.pool = torch.nn.AdaptiveAvgPool1d(128)
        
        self.dense = nn.Linear(self.final_layer, n_classes)
        
    def forward(self, x):

        out = x
        #print(out.shape)

        out = self.layers(out)
        #print(out.shape)
 
        #out = self.pool(out)
        
        #print(out.shape)
        
        out = out.view(out.size(0), -1)
        if not self.printed:
            print(out.shape)
            self.printed = True
        
        out = self.dense(out)
        
        return out
    
class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out
    
class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False, seed=0):
        super(ResNet1D, self).__init__()
        
        torch.manual_seed(seed)
        
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        out = x
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
        # out = self.do(out)
        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)
        # out = self.softmax(out)
        if self.verbose:
            print('softmax', out.shape)
        
        return out  

class WrapDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long()

    def __getitem__(self, index):
        #return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
    
    
class PyTorchModel():
    
    def __init__(self, model, n_epoch, device="cpu", batch_size=32, seed=0):
        self.model = model.to(device)  
        self.device = device
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.seed = seed
        
    def predict(self, X):
        X = X.values
        dataset = WrapDataset(X[:,None,:], np.zeros(len(X)))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        
        self.model.eval()
        all_pred_prob = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_x, input_y = tuple(t.to(self.device) for t in batch)
                pred = self.model(input_x)
                all_pred_prob.append(pred.cpu().data.numpy())
        
        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.argmax(all_pred_prob, axis=1)
        return all_pred
    
    def fit(self, X, labels):
        
        torch.manual_seed(self.seed)
        X = X.values
        class_weight = pd.Series(labels).value_counts().values
        class_weight = 1/class_weight/np.max(1/class_weight)
        print("class_weight",class_weight)
        
        ratio = 0.80
        total = len(X)
        dataset = WrapDataset(X[:int(len(X)*ratio),None,:], labels[:int(len(X)*ratio)])
        dataset_valid = WrapDataset(X[int(len(X)*ratio):,None,:], labels[int(len(X)*ratio):])

        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=self.batch_size, 
                                                 shuffle=True,
                                                 pin_memory=(self.device=="cuda"))
        dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=self.batch_size, drop_last=False)
        
        # train and test
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        loss_func = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weight).to(self.device))
        
        best = {}
        best["best_valid_score"] = 99999
        for i in range(self.n_epoch):
            self.model.train()
            losses = []
            for batch_idx, batch in enumerate(dataloader):
                input_x, input_y = tuple(t.to(self.device) for t in batch)
                pred = self.model(input_x)
                loss = loss_func(pred, input_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
#                 if (batch_idx % 10 == 0) and (len(losses) > 0):
#                     print("-",np.mean(losses))
                losses.append(loss.detach().item())

            scheduler.step(i)

            # test
            self.model.eval()
            all_pred_prob = []
            all_pred_gt = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader_valid):
                    input_x, input_y = tuple(t.to(self.device) for t in batch)
                    pred = self.model(input_x)
                    all_pred_prob.append(pred.cpu().data.numpy())
                    all_pred_gt.append(input_y.cpu().data.numpy())

            all_pred_prob = np.concatenate(all_pred_prob)
            all_pred_gt = np.concatenate(all_pred_gt)
            all_pred = np.argmax(all_pred_prob, axis=1)

            bacc = sklearn.metrics.balanced_accuracy_score(all_pred_gt,all_pred)

            print("loss",np.mean(losses), "valid_bacc",bacc)
            
            if (best["best_valid_score"] > bacc):
                best["best_model"] = self.model.state_dict()
                best["best_valid_score"] = bacc

        self.model.load_state_dict(best["best_model"])

