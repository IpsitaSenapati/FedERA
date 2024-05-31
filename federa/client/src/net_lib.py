import os
import time
from copy import deepcopy
from math import ceil
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from .data_utils import distributionDataloader
from .get_data import  get_data
# DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# #device id  of this should be same in client_lib device

def load_data(config):
    trainset, testset = get_data(config)
    # Data distribution for non-custom datasets
    if config['dataset'] == 'Sentiment140':
        trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
        testloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False)
        num_examples = {"trainset": len(trainset), "testset": len(testset)}

    if config['dataset'] != 'CUSTOM':
        datasets = distributionDataloader(config,  trainset, config['datapoints'], config['client_idx'])
        trainloader = DataLoader(datasets, batch_size= config['batch_size'], shuffle=True)
        testloader = DataLoader(testset, batch_size=config['batch_size'])
        num_examples = {"trainset": len(datasets), "testset": len(testset)}
    else:
        trainloader = DataLoader(trainset, batch_size= config['batch_size'], shuffle=True)
        testloader = DataLoader(testset, batch_size=config['batch_size'])
        num_examples = {"trainset": len(trainset), "testset": len(testset)}

    # Return data loaders and number of examples in train and test datasets
    return trainloader, testloader, num_examples


def flush_memory():
    torch.cuda.empty_cache()

def train_model(net, trainloader, epochs, device, deadline=None):
    """
    Trains a neural network model on a given dataset using SGD optimizer with Cross Entropy Loss criterion.
    Args:
        net: neural network model
        trainloader: PyTorch DataLoader object for training dataset
        epochs: number of epochs to train the model
        deadline: optional deadline time for training

    Returns:
        trained model with the difference between trained model and the received model
    """
    if net.__class__.__name__ == 'ComplexNN':
        criterion = torch.nn.BCELoss() #binary classification in sentiment analysis
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    x = deepcopy(net)
    net.train()

    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            if net.__class__.__name__ == 'ComplexNN':
                labels = labels.view(-1, 1).float()  # Reshape labels for BCELoss

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if deadline and time.time() >= deadline:
            print("Deadline occurred.")
            break

    for param_net, param_x in zip(net.parameters(), x.parameters()):
        param_net.data = param_net.data - param_x.data

    return net

def train_fedavg(net, trainloader, epochs, device, deadline=None):
    """
    Trains a given neural network using the Federated Averaging (FedAvg) algorithm.
    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model
    """
    if net.__class__.__name__ == 'ComplexNN':
        criterion = torch.nn.BCELoss() #binary classification in sentiment analysis
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.train()

    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            if net.__class__.__name__ == 'ComplexNN':
                labels = labels.view(-1, 1).float()  # Reshape labels for BCELoss

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if deadline and time.time() >= deadline:
            print("Deadline occurred.")
            break

    return net


def train_feddyn(net, trainloader, epochs, device, deadline=None, prev_grads=None):
    """
    Trains a given neural network using the FedDyn algorithm.
    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model
    """
    x = deepcopy(net)

    if net.__class__.__name__ == 'ComplexNN':
        criterion = torch.nn.BCELoss() #binary classification in sentiment analysis
        optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
        alpha = 0.01
    else:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        alpha = 0.01

    if prev_grads is not None:
        prev_grads = prev_grads.to(device)
    else:
        prev_grads = torch.cat([torch.zeros_like(param.view(-1)) for param in net.parameters()], dim=0).to(device)

    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            if net.__class__.__name__ == 'ComplexNN':
                labels = labels.float().view(-1, 1) # Reshape labels for BCELoss
            loss = criterion(outputs, labels)

            curr_params = torch.cat([param.view(-1) for param in net.parameters()], dim=0)
            lin_penalty = torch.sum(curr_params * prev_grads)
            loss -= lin_penalty

            quad_penalty = sum(torch.nn.functional.mse_loss(param, x_param, reduction='sum') for param, x_param in zip(net.parameters(), x.parameters()))
            loss += (alpha / 2) * quad_penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
            optimizer.step()
        if deadline and time.time() >= deadline:
            print("Deadline occurred.")
            break

    delta = torch.cat([torch.sub(param.data.view(-1), x_param.data.view(-1)) for param, x_param in zip(net.parameters(), x.parameters())], dim=0)
    prev_grads = torch.sub(prev_grads, delta, alpha=alpha)

    return net, prev_grads


def train_mimelite(net, state, trainloader, epochs, device, deadline=None):
    """
    Trains a given neural network using the MimeLite algorithm.
    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model
    """
    x = deepcopy(net)

    if net.__class__.__name__ == 'ComplexNN':
        criterion = torch.nn.BCELoss() #binary classification in sentiment analysis
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.train()

    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            if net.__class__.__name__ == 'ComplexNN':
                labels = labels.float().view(-1, 1) # Reshape labels for BCELoss
            loss = criterion(outputs, labels)

            grads = torch.autograd.grad(loss, net.parameters())
            with torch.no_grad():
                for param, grad, s in zip(net.parameters(), grads, state):
                    param.data -= 0.001 * ((1 - 0.9) * grad.data + 0.9 * s.to(device).data)
        if deadline and time.time() >= deadline:
            print("Deadline occurred.")
            break

    data = DataLoader(trainloader.dataset, batch_size=len(trainloader.dataset), shuffle=True)
    for images, labels in data:
        images, labels = images.to(device), labels.to(device)
        outputs = x(images)
        if net.__class__.__name__ == 'ComplexNN':
            labels = labels.float().view(-1, 1)  # Reshape labels for BCELoss
        loss = criterion(outputs, labels)
        gradient_x = torch.autograd.grad(loss, x.parameters())

    return net, gradient_x


def train_mime(net, state, control_variate, trainloader, epochs, device, deadline=None):
    """
    Trains a given neural network using the Mime algorithm.
    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model
    """
    x = deepcopy(net)

    if net.__class__.__name__ == 'ComplexNN':
        criterion = torch.nn.BCELoss() #binary classification in sentiment analysis
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.train()
    x.train()

    for epoch in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            if net.__class__.__name__ == 'ComplexNN':
                labels = labels.float().view(-1, 1)  # Reshape labels for BCELoss
            loss = criterion(outputs, labels)
            grads_y = torch.autograd.grad(loss, net.parameters())

            if epoch == 0:
                outputs_x = x(images)
                if net.__class__.__name__ == 'ComplexNN':
                    labels = labels.float().view(-1, 1)   # Reshape labels for BCELoss 
                loss_x = criterion(outputs_x, labels)
                grads_x = torch.autograd.grad(loss_x, x.parameters())
                control_variate = [gy - gx + cv for gy, gx, cv in zip(grads_y, grads_x, control_variate)]
            else:
                control_variate = [gy - gx + cv for gy, gx, cv in zip(grads_y, control_variate, state)]

            with torch.no_grad():
                for param, grad, s in zip(net.parameters(), control_variate, state):
                    param.data -= 0.001 * ((1 - 0.9) * grad.data + 0.9 * s.to(device).data)

        if deadline and time.time() >= deadline:
            print("Deadline occurred.")
            break

    data = DataLoader(trainloader.dataset, batch_size=len(trainloader.dataset), shuffle=True)
    for images, labels in data:
        images, labels = images.to(device), labels.to(device)
        outputs = x(images)
        if net.__class__.__name__ == 'ComplexNN':
            labels = labels.float().view(-1, 1) # Reshape labels for BCELoss
        loss = criterion(outputs, labels)
        gradient_x = torch.autograd.grad(loss, x.parameters())

    return net, control_variate


def train_scaffold(net, server_c, trainloader, epochs, device, deadline=None):
    """
    Trains a given neural network using the Scaffold algorithm.

    Args:
    net: A PyTorch neural network model
    server_c: Control variates from the server
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model and delta control variates
    """
    x = deepcopy(net)
    client_c = deepcopy(server_c)

    if net.__class__.__name__ == 'ComplexNN':
        criterion = torch.nn.BCELoss() #binary classification in sentiment analysis
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    lr = 0.001

    net.train()
    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            if net.__class__.__name__ == 'ComplexNN':
                labels = labels.float().view(-1, 1)  # Reshape labels for BCELoss

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Compute (full-batch) gradient of loss with respect to net's parameters
            grads = torch.autograd.grad(loss, net.parameters())

            # Update net's parameters using gradients, client_c and server_c [Algorithm line no:10]
            for param, grad, s_c, c_c in zip(net.parameters(), grads, server_c, client_c):
                s_c, c_c = s_c.to(device), c_c.to(device)
                param.data -= lr * (grad.data + (s_c.data - c_c.data))

            if deadline:
                current_time = time.time()
                if current_time >= deadline:
                    print("deadline occurred.")
                    break

    delta_c = [torch.zeros_like(param) for param in net.parameters()]
    new_client_c = deepcopy(delta_c)

    # Update net to get the difference from the initial state
    for param_net, param_x in zip(net.parameters(), x.parameters()):
        param_net.data -= param_x.data

    a = (ceil(len(trainloader.dataset) / trainloader.batch_size) * epochs * lr)
    for n_c, c_l, c_g, diff in zip(new_client_c, client_c, server_c, net.parameters()):
        c_l = c_l.to(device)
        c_g = c_g.to(device)
        n_c.data += c_l.data - c_g.data - diff.data / a

    # Calculate delta_c which equals to new_client_c - client_c
    for d_c, n_c_l, c_l in zip(delta_c, new_client_c, client_c):
        d_c = d_c.to(device)
        c_l = c_l.to(device)
        d_c.data.add_(n_c_l.data - c_l.data)

    return net, delta_c



def test_model(net, testloader, device):
    """Evaluate the performance of a model on a test dataset.

    Args:
    net (torch.nn.Module): The neural network model to evaluate.
    testloader (torch.utils.data.DataLoader): The data loader for the test dataset.
    device (torch.device): The device to run the evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
    Tuple: The average loss and accuracy of the model on the test dataset.
    """
    if net.__class__.__name__ == 'ComplexNN':
        criterion = torch.nn.BCELoss() #binary classification in sentiment analysis
    else:
        criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    test_loss, correct, total = 0.0, 0, 0  
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)
            if net.__class__.__name__ == 'ComplexNN':
                labels = labels.view(-1, 1).float()      # Reshape labels for BCELoss   
            outputs = net(images)
            test_loss += criterion(outputs, labels).item()
            if net.__class__.__name__ == 'ComplexNN':
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
            else:
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()           
            total += labels.size(0)        
        test_loss /= len(testloader)  
        accuracy = correct / total

    return test_loss, accuracy
