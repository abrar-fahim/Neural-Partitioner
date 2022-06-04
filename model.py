import torch
import torch.nn as nn
import argparse
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loader import MyDataset
from loss_fn import MyLoss
import matplotlib.pyplot as plt


class LinearModel(nn.Module):

    
    def __init__(self, n_input, n_hidden, num_class, opt, toplevel=False):

        super(LinearModel, self).__init__()

        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.main_device="cuda:0" if torch.cuda.is_available() else "cpu"
        self.secondary_device="cuda:0"



        self.n_hidden = n_hidden

        print('self n class ', num_class)
        self.n_classes = num_class



        self.fc1 = nn.Linear(n_input, num_class)
        self.fc_confidence = nn.Linear(n_input, 1)


        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)

        self.n_input = n_input
        self.n_classes = num_class
        self.id = None # used in file naming

        # self.double()



    def forward(self, x):

        x = x.float()


        


        bins_prediction = self.fc1(x)
        bins_prediction = self.softmax(bins_prediction)

        confidence = self.fc_confidence(x)
        confidence =  self.sigmoid(confidence)



        return (bins_prediction, confidence.flatten())




class NeuralModel(nn.Module):

    
    def __init__(self, n_input, n_hidden, num_class, opt, toplevel=False):

        super(NeuralModel, self).__init__()

        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.main_device="cuda:0" if torch.cuda.is_available() else "cpu"
        self.secondary_device="cuda:0"


    

        self.n_hidden = n_hidden

        print('self n class ', num_class)
        self.n_classes = num_class

        self.dropout = nn.Dropout(p=0.1)


        self.block1 = self.net_block(n_input, self.n_hidden)

        self.block2 = self.net_block(self.n_hidden, self.n_hidden)


        self.fc1 = nn.Linear(self.n_hidden, num_class)

        self.softmax = nn.Softmax(dim=-1)


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)

        self.n_input = n_input
        self.n_classes = num_class
        self.id = None 



    def net_block(self, n_in, n_out):

        block = nn.Sequential(nn.Linear(n_in, n_out),
                            # nn.BatchNorm1d(n_out, track_running_stats=False),
                            nn.BatchNorm1d(n_out),
                            nn.ReLU())
        return block


    def forward(self, x):

        x = x.float()

        y = self.block1(x)


        y = self.block2(y)  

        y = self.dropout(y)

        


        bins_prediction = self.fc1(y)
        bins_prediction = self.softmax(bins_prediction)
  


        confidence = torch.max(bins_prediction, dim=1)[0]


        return (bins_prediction, confidence.flatten())
