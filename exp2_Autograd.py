#! /usr/bin/python

# How does PyTorch compute differentiation?

import torch

# create a tensor with requires_grad is True
x = torch.ones(2, 2, requires_grad=True)
# can also modify requires_grad later
# x = torch.ones(2, 2)
# x.requires_grad_()
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

# can we compute gradient of a non-scalar?
