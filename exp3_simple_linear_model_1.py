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
        self.layer = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=inter_features, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=inter_features, out_features=out_features, bias=bias)
        )
        self.reset_parameters()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        # the shape of x should be batch_size x in_features
        x = self.layer(x)
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
            model.train()
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
        model.eval()
        out_test = model(test_input)
        loss_test = criterion(out_test, test_target)
        print("epoch %d: test loss %.4f" % (epoch, loss.item()))
