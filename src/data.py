from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
