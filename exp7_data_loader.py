import cv2
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as utils
from logger import Logger
from tensorboardX import SummaryWriter

# reference: http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# inherit from torch.utils.data.Dataset and overwirte __len__ & __getitem__
class myDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        for __, __, filenames in os.walk(self.root):
            self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    # read the images in __getitem__
    # this is memory efficient because all the images are not stored in the memory at once but read as required
    def __getitem__(self, index):
        filename = self.filenames[index]
        image = cv2.imread(self.root + '/' + filename)
        image = cv2.resize(image, (320,320), cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        data = {
            'name': filename,
            'data': image
        }
        return data

if __name__ == "__main__":
    logger = Logger('./data_loader_logs')
    # writer = SummaryWriter('./data_loader_logs')
    batch_size = 6
    # instantiate torch.utils.data.DataLoader
    transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    imgLoader = data.DataLoader(
            myDataset(root="faces", transform=transform),
            batch_size=batch_size, shuffle=True, num_workers=2)
    for i, data in enumerate(imgLoader, 0):
        print("[%d]\n" % i)
        print(data['name']);
        print('\n')

        info = {
            'images': data['data'][:3]
        }
        for tag, images in info.items():
            logger.image_summary(tag, images, i)
        # images = utils.make_grid(data['data'], nrow=2)
        # writer.add_image('Image', images, i)
