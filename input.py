import glob
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

face_dataset = datasets.ImageFolder('train', transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(face_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)

for i, (input, target) in enumerate(dataset_loader):
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

# reference : https://github.com/pytorch/examples/blob/master/imagenet/main.py#L97-L121

"""
loader = transforms.Compose([
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image

"""
