import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

class LinearModel(nn.Module):
    def __init__(self, in_features, inter_features, out_features, bias=True):
        super(LinearModel, self).__init__()
        self.bias = bias
        self.training = True
        self.weight_param1 = nn.Parameter(
            torch.FloatTensor(inter_features, in_features).normal_(0.0, 0.01)
        )
        self.weight_param2 = nn.Parameter(
            torch.FloatTensor(out_features, inter_features).normal_(0.0, 0.01)
        )
        if self.bias:
            self.bias_param1 = nn.Parameter(
                torch.FloatTensor(inter_features).fill_(0.)
            )
            self.bias_param2 = nn.Parameter(
                torch.FloatTensor(out_features).fill_(0.)
            )
    def set_mode(self, training):
        self.training = training
    def forward(self, x):
        # the shape of x should be batch_size x in_features
        x = F.linear(input=x, weight=self.weight_param1,
            bias=self.bias_param1 if self.bias else None)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.linear(input=x, weight=self.weight_param2,
            bias=self.bias_param2 if self.bias else None)
        return x

if __name__ == "__main__":
    # define the model
    model = LinearModel(64, 32, 4, True)
    model = model.to(device)

    # loss function
    criterion = nn.MSELoss()

    # data, batch_size = 1000
    input_data  = torch.randn(1000, 64).to(device)
    target_data = torch.randn(1000, 4).to(device)
    test_input  = torch.randn(200, 64).to(device)
    test_target = torch.randn(200, 4).to(device)

    # optimizer
    optimizer = optim.SGD(params=model.parameters(), lr=0.01)

    # train for 100 epochs
    for epoch in range(100): # iterate through epochs
        torch.cuda.empty_cache()
        for i in range(1000): # iterate through dataset
            # set to training mode
            model.set_mode(True)
            # clear grad cache
            model.zero_grad()
            optimizer.zero_grad()
            # forward
            input  = input_data[i,:].unsqueeze(0)
            target = target_data[i,:].unsqueeze(0)
            out = model(input)
            loss = criterion(out, target)
            # backward
            loss.backward()
            optimizer.step()
        #  after each epoch: check loss on test set
        model.set_mode(False)
        with torch.no_grad():
            out_test = model(test_input)
            loss_test = criterion(out_test, test_target)
            print("epoch %d: test loss %.4f" % (epoch, loss.item()))
