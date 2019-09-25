import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

class Classification_data(Dataset):
    def __init__(self, data, label=None):
        super(Classification_data, self).__init__()

        self.data = data
        self.label = label

    def __getitem__(self, index):

        if self.label is None:
            return self.data[index]
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

class MLP(nn.Module):
    def __init__(self, input_size,
                 layers_size,
                 num_labels,
                 batchnorm= True,
                 dropout_rate= 0.15,
                 activation = 'relu'
                 ):
        super(MLP, self).__init__()

        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.layers_size = layers_size
        self.num_labels = num_labels
        #self.layers_size = layers_size.insert(0, input_size)

        model = []
        for layer_idx, units in enumerate(self.layers_size):
            if layer_idx == len(self.layers_size)-1:
                model.append(nn.Linear(self.layers_size[layer_idx], self.num_labels))
                model.append(nn.Sigmoid())
                break

            model.append(nn.Linear(self.layers_size[layer_idx], self.layers_size[layer_idx + 1]))
            model.append(nn.ReLU())
            if self.batchnorm:
                model.append(nn.BatchNorm1d(self.layers_size[layer_idx + 1]))
            model.append(nn.Dropout(self.dropout_rate))
        self.layers = nn.Sequential(*model)

    def forward(self, x):
        return self.layers(x.type(torch.FloatTensor))

class MLP_train():

    def __init__(self, 
                 net,
                 epochs=10,
                 optimizer='Adam',
                 momentum= 0.9,
                 lr = 0.001,
                 num_labels= 3):

        assert optimizer in ['Adam', 'SGD']
        self.lr = lr
        self.net= net
        self.epochs = epochs
        self.momentum = momentum
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.num_labels = num_labels

        return

    def init_weights(layer):
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform(layer.weight)
            layer.bias.data.fill_(0.01)
    def label_mapping(self,labels):
        self.label_map = list( zip( set(labels), [i for i in range(len(set(labels)))]))
        return
    def label2idx(self, labels):
        d = dict(self.label_map)
        return list(map(lambda x:d[x], labels))
        return 
    def idx2label(self, idx):
        return list(map(lambda x: self.label_map[x][0], idx))


    def fit(self,x_train,
                 y_train):

        dtype = torch.float
        #device = torch.device("gpu")

        label = np.asarray(y_train).astype(np.int_)
        #label[label==4]= 3 
        #label_onehot = (np.arange(self.num_labels) == label_onehot[:,None]).astype(np.int_)
        #label = torch.from_numpy(label-1 )
        self.label_mapping(label)
        label = self.label2idx(label)
        dataset = Classification_data(x_train, label)
        data_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64, shuffle=True,
                                             num_workers=5)
        criterion = nn.CrossEntropyLoss()
        optimize = getattr(optim,self.optimizer)(self.net.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            running_loss = 0
            i=0
            for data, label in data_loader:
                # zero the parameter gradients
                optimize.zero_grad()
                output = self.net(data)
                loss = criterion(output, label)
                loss.backward()
                optimize.step()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print('Finished Training')
        return self

    def predict(self, x_test):
        correct = 0
        total = 0
        dataset = Classification_data(x_test)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=250, num_workers=5, shuffle = False)
        predicted_list = []
        with torch.no_grad():
            for data in testloader:
                outputs = self.net(data)
                _, predicted = torch.max(outputs.data,1)
                #return the labels to their initial form
                predicted = torch.IntTensor(self.idx2label(predicted))
                #predicted[predicted==3] = 4
                predicted_list.extend(predicted.numpy().tolist())
        return predicted_list

