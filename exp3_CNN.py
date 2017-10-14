#! /usr/bin/python

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # define conv operations
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                            kernel_size=5, stride=1, padding=0,
                            dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # define fc operations
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120, bias=True)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=None, padding=0)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.num_flat_features(x)) # refer to: http://pytorch.org/docs/master/tensors.html#torch.Tensor.view
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == "__main__":
    ## define the network
    # net = Net()

    # print net
    ## learnable parameters are returned by net.parameters()
    # params = list(net.parameters())
    # print(params[0].size())  # size of conv1's weight
    # print(params[1].size())  # size of conv1's bias

    ## forward
    # the input to the forward is an autograd.Variable which has size [batchSize,numChannels,height,width]
    # input = Variable(torch.randn(1, 1, 32, 32))
    # out = net(input)
    # print(out)

    ## backward
    # zero the gradient buffers of all parameters and backprops with random gradients
    # net.zero_grad()
    # out.backward(torch.randn(1, 10))

    ## compute the loss
    # target = Variable(torch.arange(1, 11))
    # criterion = nn.MSELoss()
    # loss = criterion(out, target)

    ## take all things together and perform backpropogation
    net = Net()
    net.cuda()
    input = Variable(torch.randn(1, 1, 32, 32).cuda())
    target = Variable(torch.arange(1, 11).cuda())
    criterion = nn.MSELoss()
    optimizer = optim.SGD(params=net.parameters(), lr=0.01)
    for i in range(300):
        optimizer.zero_grad()
        out = net(input)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print ("loss: %f" % loss.data[0])
