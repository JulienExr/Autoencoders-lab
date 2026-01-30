from sklearn.preprocessing import normalize
import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision import datasets, transforms
import kagglehub


def get_mnist_dataloaders(batch_size=64, normalize=False):
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.ToTensor()
    
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    mnist_dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False)

    return dataloader, test_dataloader

def get_fashion_mnist_dataloaders(batch_size=64, normalize=False):
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.ToTensor()
    
    fashion_mnist_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(fashion_mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    fashion_mnist_dataset_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(fashion_mnist_dataset_test, batch_size=batch_size, shuffle=False)

    return dataloader, test_dataloader



def get_cifar10_dataloaders(batch_size=64, normalize=False):
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.ToTensor()
    
    cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    cifar10_dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(cifar10_dataset_test, batch_size=batch_size, shuffle=False)

    return dataloader, test_dataloader






















#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
def add_noise(img, sigma):
    sigma = sigma / 255.0
    noise = torch.randn_like(img) * sigma
    noisy_img = img + noise
    noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
    return noisy_img


class LandscapeDenoisingDataset(Dataset):
    def __init__(self, root_dir, transforms, sigma=25):
        self.root_dir = root_dir
        self.sigma = sigma

        self.images = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.transforms(img)

        noisy = add_noise(img, self.sigma)

        return noisy, img


def get_landscape_dataloaders(batch_size=256, normalize=True):

    path = kagglehub.dataset_download("arnaud58/landscape-pictures")

    print("Path to dataset files:", path)

    if normalize:
        transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transforms = transforms.ToTensor()

    dataset = LandscapeDenoisingDataset(root_dir=path, transforms=transforms, sigma=25)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader