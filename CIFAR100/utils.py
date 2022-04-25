from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR100

from random import sample 
import cv2
import numpy as np

class CIFAR100Pair_true_label(CIFAR100):
    #dataloader where pairs of positive samples are randomly sampled from pairs
    #of inputs with the same label. 
    def __init__(self, root='../data', train=True, transform=None):
        super().__init__(root=root, train=train, transform=transform)
        def get_labels(i):
            return [index for index in range(len(self)) if self.targets[index]==i]

        self.label_index = [get_labels(i) for i in range(100)]

    def __getitem__(self, index):
        img1, target = self.data[index], self.targets[index]

        index_example_same_label=sample(self.label_index[self.targets[index]],1)[0]
        img2 = self.data[index_example_same_label]

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        if self.transform is not None:
            pos_1 = self.transform(img1)
            pos_2 = self.transform(img2)
 
        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target, index

class CIFAR100Pair(CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
 
        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target, index

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=int(0.1 * 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

def get_dataset(dataset_name, dataset_location, pair=True, return_index=False):

    if pair:
        if dataset_name=='cifar100':
            train_data = CIFAR100Pair(root=dataset_location, train=True, transform=train_transform,download=True)
            memory_data = CIFAR100Pair(root=dataset_location, train=True, transform=test_transform,download=True)
            test_data = CIFAR100Pair(root=dataset_location, train=False, transform=test_transform,download=True)
        elif dataset_name=='cifar100_true_label':
            train_data = CIFAR100Pair_true_label(root=dataset_location, train=True, transform=train_transform)
            memory_data = CIFAR100Pair_true_label(root=dataset_location, train=True, transform=test_transform)
            test_data = CIFAR100Pair_true_label(root=dataset_location, train=False, transform=test_transform)
        else:
            raise Exception('Invalid dataset name')
    else:
        train_tx_to_use = train_transform
        try:
            if dataset_name in ['cifar100', 'cifar100_true_label']:
                train_data = CIFAR100(root=dataset_location, train=True, transform=train_tx_to_use)
                memory_data = CIFAR100(root=dataset_location, train=True, transform=test_transform)
                test_data = CIFAR100(root=dataset_location, train=False, transform=test_transform)
            else:
                raise Exception('Invalid dataset name')
        except:
            if dataset_name in ['cifar100', 'cifar100_true_label']:
                train_data = CIFAR100(root=dataset_location, train=True, transform=train_tx_to_use)
                memory_data = CIFAR100(root=dataset_location, train=True, transform=test_transform)
                test_data = CIFAR100(root=dataset_location, train=False, transform=test_transform)
            else:
                raise Exception('Invalid dataset name')


    return train_data, memory_data, test_data
        
