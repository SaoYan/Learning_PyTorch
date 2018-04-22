#! /usr/bin/python

# How does PyTorch compute differentiation?

import torch
from torch.autograd import Variable

# create a varible
x = Variable(torch.ones(2, 2), requires_grad=True)
print("\nx:")
print(x)

# series of operations
y = x + 2
print("\ny:")
print(y)

z = y * y * 3
print("\nz:")
print(z)

out = z.mean()
print("\nout:")
print(out)

# backprop
out.backward()

# gradients (we expect 4.5)
print("\nx.grad:")
print(x.grad)
