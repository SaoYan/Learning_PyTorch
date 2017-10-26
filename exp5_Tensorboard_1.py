#! /usr/bin/python

# Use tensorboard visualization toolkit in PyTorch
# reference: https://zhuanlan.zhihu.com/p/27624517
# cannot add graph, embeddings

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from logger import Logger

transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./CIFAR10_data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./CIFAR10_data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def to_np(x):
    return x.data.cpu().numpy()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*5*5, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*5*5)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    net = Net()
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # Set the logger
    logger = Logger('./CIFAR10_logs')
    # training
    step = 0
    for epoch in range(1): # loop over the dataset multiple times; not the # steps!
        running_loss = 0.0
        correct = 0
        total = 0
        for i,data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            if (i+1) % 100 == 0:
                # trainging loss
                print('[%d, %d] loss: %f' %(epoch+1, i+1, running_loss/100))
                # test accuracy
                for data in testloader:
                    images_test, labels_test = data
                    images_test, labels_test = Variable(images_test.cuda()), labels_test.cuda()
                    outputs_test = net(images_test)
                    _, predicted = torch.max(outputs_test.data, 1)
                    total += labels_test.size(0)
                    correct += (predicted == labels_test).sum()
                print("test accuracy: %d%%\n" % (100*correct/total))

                #============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'loss': running_loss/100,
                    'accuracy': 100*correct/total
                }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step)
                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), step)
                    logger.histo_summary(tag+'/grad', to_np(value.grad), step)
                # (3) Log the images
                info = {
                    'images': to_np(inputs.view(-1,3,32,32)[:3])
                }
                for tag, images in info.items():
                    logger.image_summary(tag, images, step)
                # reset
                running_loss = 0.0
                correct = 0
                total = 0
                step += 1
