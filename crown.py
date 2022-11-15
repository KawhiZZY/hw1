"""
   crown.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self.layer1 = nn.Linear(2, hid)
        self.layer2 = nn.Linear(hid, hid)
        self.layer3 = nn.Linear(hid, 1)

    def forward(self, input):
        self.hid1 = torch.tanh(self.layer1(input))
        self.hid2 = torch.tanh(self.layer2(self.hid1))
        output = torch.sigmoid(self.layer3(self.hid2))
        return output

class Full4Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full4Net, self).__init__()
        self.layer1 = nn.Linear(2, hid)
        self.layer2 = nn.Linear(hid, hid)
        self.layer3 = nn.Linear(hid, hid)
        self.layer4 = nn.Linear(hid, 1)

    def forward(self, input):
        self.hid1 = torch.tanh(self.layer1(input))
        self.hid2 = torch.tanh(self.layer2(self.hid1))
        self.hid3 = torch.tanh(self.layer3(self.hid2))
        output = torch.sigmoid(self.layer4(self.hid3))
        return output

class DenseNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()
        self.layer1 = nn.Linear(2, num_hid)
        self.layer2 = nn.Linear(num_hid+2, num_hid)
        self.layer3 = nn.Linear(num_hid+num_hid+2, 1)

    def forward(self, input):
        self.hid1 = torch.tanh(self.layer1(input))
        self.hid2 = torch.tanh(self.layer2(torch.cat((self.hid1, input), 1)))
        output = torch.sigmoid(self.layer3(torch.cat((self.hid2, self.hid1, input), 1)))
        return output
