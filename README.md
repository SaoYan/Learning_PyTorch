# LearningPytorch
Simple demos for PyTorch.  
Reference: [PyTorch online tutorials](http://pytorch.org/tutorials/)

## exp1_Tensor.py:
[Reference](http://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py)      
This demo includes:
1. How to construct a Tensor in PyTorch
2. Basic operations that can be applied to Tensor objects
3. How to convert a PyTorch-Tensor to Numpy-Ndarray, and vice versa.
4. How to move Tensors onto GPU for efficient computation.

## exp2_Variable_and_Autograd.py:
[Reference](http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)    
This demo shows how to compute differentiation for all operations on Tensors. PyTorch uses ***autograd*** package.

## exp3_CNN.py:
This demo shows how to construct a CNN in PyTorch.
1. Two convolutional layers, each of which is follows by one ReLU activation and one max-pooling layer.
2. Three fully-connected layers.

## exp4_CIFAR10_CNN.py
[Reference](http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)    
This demo applies the CNN in exp3 and trains a classifier on CIFRA10 dataset.
1. cost function: cross entropy loss    
2. SGD optimizer, learning rate 1e-4, momentum 0.9

## exp5_Tensorboard.py
One disadvantage of PyTorch is that it doesn't have its own visualization toolkit. But it is possible to use Tensorboard in PyTorch, thanks to [tensorboardX]((https://github.com/lanpa/tensorboard-pytorch)). ***(Of course, you need to install tensorlfow first!)***  

1. Install tensorboardX
```
pip install tensorboardX
```
2. Import tensorboardX
```
from tensorboardX import SummaryWriter
```
3. Save tensorboard logs
```
# (1) Log the scalar values
writer.add_scalar('loss', running_loss/100, step)
writer.add_scalar('accuracy', 100*correct/total, step)
# (2) Log values and gradients of the parameters (histogram)
for tag, value in net.named_parameters():
    tag = tag.replace('.', '/')
    writer.add_histogram(tag, to_np(value), step)
    writer.add_histogram(tag+'/grad', to_np(value.grad), step)
# (3) Log the images
images = utils.make_grid(inputs.view(-1,3,32,32).data, nrow=5,
                                    normalize=True, scale_each=True)
writer.add_image('Image', images, step)
```

## exp6_data_loader.py
[Reference](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html)    
One important thing in practice: loading your own dataset. If you are a former Tensorflow user, you must be familiar with constructing TFrecords files. PyTorch also provides an efficient way to load data.

1. First, inherit from ***torch.utils.data.Dataset*** and overwirte ***\_\_len\_\_*** & ***\_\_getitem\_\_***
* overwirte ***\_\_len\_\_*** so that len(dataset) returns the size of the dataset.
* overwirte ***\_\_getitem\_\_*** to support the indexing such that dataset[i] can be used to get the *ith* sample
2. Senond, instantiate ***torch.utils.data.DataLoader***
3. Iterate through the entier dataset, getting one batch each time for backpropgation.
