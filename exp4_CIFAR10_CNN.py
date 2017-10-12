#! /usr/bin/python

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./CIFAR10_data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./CIFAR10_data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # define the network
    net = Net()
    net.cuda()
    # define loss
    criterion = nn.CrossEntropyLoss()
    # define optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # training
    for epoch in range(2): # loop over the dataset multiple times; not the # steps!
        running_loss = 0.0
        correct = 0
        total = 0
        for i,data in enumerate(trainloader, 0):
            # i is the # inner steps
            # training set contains 50000 images, since the batch size is 4, i will be 0~12499(12500 inner steps)
            inputs, labels = data
            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            # zero the parameter gradients, else gradients will be accumulated to existing gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            if (i+1) % 100 == 0:    # print every 2000 mini-batches
                # trainging loss
                print('[%d, %d] loss: %f' %(epoch+1, i+1, running_loss/2000))
                running_loss = 0.0
                # test accuracy
                for data in testloader:
                    images_test, labels_test = data
                    images_test, labels_test = Variable(images_test.cuda()), labels_test.cuda()
                    outputs_test = net(images_test)
                    _, predicted = torch.max(outputs_test.data, 1)
                    total += labels_test.size(0)
                    correct += (predicted == labels_test).sum()
                print("test accuracy: %d%%\n" % (100*correct/total))
                correct = 0
                total = 0
