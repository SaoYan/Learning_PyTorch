#! /usr/bin/python

# How does PyTorch compute differentiation?

import torch
from torch.autograd import Variable

# create a varible
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)
# variable x is user-defined, so it doesn't have a grad_fn
print(x.grad_fn)

# series of operations
y = x + 2
# variable y was created as a result of an operation, so it has a grad_fn
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print(out)

# backprop
out.backward()

# gradients
# d(out)/d(x) (we expect 4.5)
print(x.grad)
