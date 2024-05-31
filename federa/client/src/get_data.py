import os
import torch
from torchvision import transforms,datasets
from torch.utils import data
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import gdown

# Define a function to get the train and test datasets based on the given configuration
def get_data(config):
    # If the dataset is not custom, create a dataset folder
    if config['dataset'] != 'CUSTOM':
        dataset_path = "client_dataset"
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

    # Get the train and test datasets for each supported dataset
    if config['dataset'] == 'MNIST':
        # Apply transformations to the images
        apply_transform = transforms.Compose([transforms.Resize(config["resize_size"]), transforms.ToTensor()])
        # Download and load the trainset
        trainset = datasets.MNIST(root='client_dataset/MNIST', train=True, download=True, transform=apply_transform)
        # Download and load the testset
        testset = datasets.MNIST(root='client_dataset/MNIST', train=False, download=True, transform=apply_transform)
    elif config['dataset'] == 'FashionMNIST':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.FashionMNIST(root='client_dataset/FashionMNIST',
                                        train=True, download=True, transform=apply_transform)
        testset = datasets.FashionMNIST(root='client_dataset/FashionMNIST',
                                        train=False, download=True, transform=apply_transform)
    elif config['dataset'] == 'CIFAR10':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.CIFAR10(root='client_dataset/CIFAR10',
                                    train=True, download=True, transform=apply_transform)
        testset = datasets.CIFAR10(root='client_dataset/CIFAR10',
                                   train=False, download=True, transform=apply_transform)
    elif config['dataset'] == 'CIFAR100':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.CIFAR100(root='client_dataset/CIFAR100',
                                     train=True, download=True, transform=apply_transform)
        testset = datasets.CIFAR100(root='client_dataset/CIFAR100',
                                    train=False, download=True, transform=apply_transform)
    elif config['dataset'] == 'Sentiment140':
        output_train = './client_dataset/Sentiment_train.npy'
        output_test = './client_dataset/Sentiment_test.npy'
        '''
        To datasets used below are uploaded on google drive and can be accessed at:
        Entire dataset:-
        Trainset:https://drive.google.com/file/d/1jrqDDV9Myoralnr2hEFAuzDvzkd2RIpx/view?usp=drive_link
        Testset:https://drive.google.com/file/d/16WT66icsbmGxQSjQK-BIZ0VSPf0bVBZZ/view?usp=drive_link
        Subset of the dataset:-
        Trainset:https://drive.google.com/file/d/1g-zJMgQSCo72ZtvLlRVWoLKG99TNexPa/view?usp=drive_link
        Testset:https://drive.google.com/file/d/1dvaE5FlDj8yjExdcWa1XhQA9CGrps-vK/view?usp=drive_link
        '''
        
        #Using a subset of the dataset
        file_id_train = '1g-zJMgQSCo72ZtvLlRVWoLKG99TNexPa'
        file_id_test = '1dvaE5FlDj8yjExdcWa1XhQA9CGrps-vK'
        #to use the entire dataset, use the file_ids given below
        #file_id_train = '1jrqDDV9Myoralnr2hEFAuzDvzkd2RIpx'
        #file_id_test = '16WT66icsbmGxQSjQK-BIZ0VSPf0bVBZZ'
        url_train = f'https://drive.google.com/uc?id={file_id_train}'
        url_test = f'https://drive.google.com/uc?id={file_id_test}'
        gdown.download(url_train, output_train, quiet=False)
        gdown.download(url_test, output_test, quiet=False)
        trainset = Sentiment140Dataset(np.load(output_train, allow_pickle=True).item())
        testset = Sentiment140Dataset(np.load(output_test, allow_pickle=True).item())
    elif config['dataset'] == 'CUSTOM':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        # Load the custom dataset
        trainset = customDataset(root='client_custom_dataset/CUSTOM/train', transform=apply_transform)
        testset = customDataset(root='client_custom_dataset/CUSTOM/test', transform=apply_transform)
    else:
        # Raise an error if an unsupported dataset is specified
        raise ValueError(f"Unsupported dataset type: {config['dataset']}")


    # Return the train and test datasets
    return trainset, testset

class customDataset(data.Dataset):
    def __init__(self, root, transform=None):
        """
        Custom dataset class for loading image and label data from a folder of .npy files.
        Args:
            root (str): Path to the folder containing the .npy files.
            transform (callable, optional): A function/transform that takes
              an PIL image and returns a transformed version.
                                            E.g, `transforms.RandomCrop`
        """

        self.root = root
        samples = sample_return(root)

        self.samples = samples

        self.transform = transform

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset at the given index.
        Args:
            index (int): Index of the sample to retrieve.
        Returns:
            img (PIL.Image): The image data.
            label (int): The label for the image data.
        """
        img, label= self.samples[index]

        img = np.load(img)

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)


        return img, label

    def __len__(self):
        return len(self.samples)

def sample_return(root):
    # Initialize an empty list to hold the samples
    newdataset = []
    # Define a dictionary that maps label names to integer values
    labels = {'Breast': 0, 'Chestxray':1, 'Oct': 2, 'Tissue': 3}
    # Loop over each image in the root directory
    for image in os.listdir(root):
        # Initialize an empty list to hold the label
        label=[]
        # Get the full path of the image
        path = os.path.join(root, image)
        # Extract the label from the image filename
        labels_str = image.split('_')[0]
        label = labels[labels_str]
        # Create a tuple containing the image path and its label, and append it to the list of samples
        item = (path, label)
        newdataset.append(item)
    # Return the list of samples
    return newdataset

class Sentiment140Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = torch.tensor(data['data'], dtype=torch.float)
        self.labels = torch.tensor(data['target'], dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            text = self.transform(text)

        return text, label
