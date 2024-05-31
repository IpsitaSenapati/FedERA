import torch
import os
from tqdm import tqdm
from torchvision import transforms,datasets
from torch.utils.data import DataLoader,Dataset
from torch import nn
from torchvision import models
from torch.utils import data
import numpy as np
from PIL import Image
import gdown

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#serverlib and eval_lib should be on the same device

def load_data(config):
    testset, _ = get_data(config)
    testloader = DataLoader(testset, batch_size=config['batch_size'])
    num_examples = {"testset": len(testset)}
    return testloader, num_examples

### Load different dataset
def get_data(config):
    dataset_path="./server_dataset"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    if config['dataset'] == 'MNIST':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        testset = datasets.MNIST(root='./server_dataset/MNIST',
                                train=False, download=True, transform=apply_transform)
        trainset = datasets.MNIST(root='./server_dataset/MNIST',
                                train=True, download=True, transform=apply_transform)
    if config['dataset'] == 'FashionMNIST':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        testset = datasets.FashionMNIST(root='./server_dataset/FashionMNIST',
                                        train=False, download=True, transform=apply_transform)
        trainset = datasets.FashionMNIST(root='./server_dataset/FashionMNIST',
                                        train=True, download=True, transform=apply_transform)

    if config['dataset'] == 'CIFAR10':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        testset = datasets.CIFAR10(root='./server_dataset/CIFAR10',
                                   train=False, download=True, transform=apply_transform)
        trainset = datasets.CIFAR10(root='./server_dataset/CIFAR10',
                                   train=True, download=True, transform=apply_transform)

    if config['dataset'] == 'CIFAR100':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        testset = datasets.CIFAR100(root='./server_dataset/CIFAR100',
                                    train=False, download=True, transform=apply_transform)
        trainset = datasets.CIFAR100(root='./server_dataset/CIFAR100',
                                    train=True, download=True, transform=apply_transform)
    if config['dataset'] == 'Sentiment140':
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
        gdown.download(url_train, output_train, quiet=False)
        gdown.download(url_test, output_test, quiet=False)
        url_train = f'https://drive.google.com/uc?id={file_id_train}'
        url_test = f'https://drive.google.com/uc?id={file_id_test}'
        trainset = Sentiment140Dataset(np.load(output_train, allow_pickle=True).item())
        testset = Sentiment140Dataset(np.load(output_test, allow_pickle=True).item())
    if config['dataset'] == 'CUSTOM':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        testset = customDataset(root='./server_custom_dataset/CUSTOM/test', transform=apply_transform)
        trainset = customDataset(root='./server_custom_dataset/CUSTOM/train', transform=apply_transform)

    return testset, trainset

class customDataset(data.Dataset):
    def __init__(self, root, transform=None):

        self.root = root
        samples = sample_return(root)

        self.samples = samples

        self.transform = transform

    def __getitem__(self, index):
        img, label= self.samples[index]

        img = np.load(img)

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)


        return img, label

    def __len__(self):
        return len(self.samples)
    
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
    
def sample_return(root):
    newdataset = []
    labels = {'Breast': 0, 'Chestxray':1, 'Oct': 2, 'Tissue': 3}
    for image in os.listdir(root):
        label=[]
        #print(image)
        path = os.path.join(root, image)
        #print(path)
        labels_str = image.split('_')[0]
        label = labels[labels_str]
        item = (path, label)
        newdataset.append(item)
    return newdataset

class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 400)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.logSoftmax(x)
        return x

class ComplexNN(nn.Module):
    def __init__(self, input_dim):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x
    
def get_net(config):
    if config["net"] == 'LeNet':
        if config['dataset'] in ['MNIST', 'FashionMNIST', 'CUSTOM']:
            net = LeNet(in_channels=1, num_classes=10)
        elif config['dataset'] == 'CIFAR10':
            net = LeNet(in_channels=3, num_classes=10)
        else:
            net = LeNet(in_channels=3, num_classes=100)
    if config["net"] == 'resnet18':
        if config['dataset'] == 'CIFAR10':
            net = models.resnet18(num_classes=10)
        else:
            net = models.resnet18(num_classes=100)
    if config["net"] == 'resnet50':
        if config['dataset'] == 'CIFAR10':
            net = models.resnet50(num_classes=10)
        else:
            net = models.resnet50(num_classes=100)
    if config["net"] == 'vgg16':
        if config['dataset'] == 'CIFAR10':
            net = models.vgg16(num_classes=10)
        else:
            net = models.vgg16(num_classes=100)
    if config['net'] == 'AlexNet':
        if config['dataset'] == 'CIFAR10':
            net = models.alexnet(num_classes=10)
        else:
            net = models.alexnet(num_classes=100)
    if config['net'] == 'ComplexNN':
        net=ComplexNN(input_dim=300)
    return net

def train_model(net, trainloader):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for data in trainloader:
        inputs, labels = data
        if isinstance(net, ComplexNN):
            criterion = torch.nn.BCELoss()
            inputs = inputs.float() 
            labels = labels.float().unsqueeze(1)
        else:
            criterion = torch.nn.CrossEntropyLoss()
            inputs = inputs.float()  
            labels = labels.long()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return net

def test_model(net, testloader, device):
    criterion = torch.nn.BCELoss() if isinstance(net, ComplexNN) else torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in tqdm(testloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            if isinstance(net, ComplexNN):
                inputs = inputs.float()  
                labels = labels.float().unsqueeze(1)  
            else:
                inputs = inputs.float()  
                labels = labels.long()  

            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            if isinstance(net, ComplexNN):
                predicted = (outputs >= 0.5).float()  # For binary classification
                correct += (predicted == labels).sum().item()
            else:
                _, predicted = torch.max(outputs.data, 1)  # For multi-class classification
                correct += (predicted == labels).sum().item()

            total += labels.size(0)

    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def save_intial_model(config):
    testloader, _ = load_data(config)
    net = get_net(config)
    net = train_model(net, testloader)
    torch.save(net.state_dict(), 'initial_model.pt')
