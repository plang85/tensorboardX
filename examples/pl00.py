import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter
import datetime

writer = SummaryWriter()

npoints = 3000
dataset = datasets.MNIST('mnist', train=False, download=True)
images = dataset.test_data[:npoints].float()
label = dataset.test_labels[:npoints]
features = images.view(npoints, 784)
#writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))
writer.add_embedding(features, label_img=images.unsqueeze(1))
"""
writer.add_embedding(features, global_step=1, tag='noMetadata')

dataset = datasets.MNIST('mnist', train=True, download=True)
images_train = dataset.train_data[:100].float()
labels_train = dataset.train_labels[:100]
features_train = images_train.view(100, 784)

all_features = torch.cat((features, features_train))
all_labels = torch.cat((label, labels_train))
all_images = torch.cat((images, images_train))
dataset_label = ['test'] * 100 + ['train'] * 100
all_labels = list(zip(all_labels, dataset_label))

writer.add_embedding(all_features, metadata=all_labels, label_img=all_images.unsqueeze(1),
                     metadata_header=['digit', 'dataset'], global_step=2)
"""

writer.close()
