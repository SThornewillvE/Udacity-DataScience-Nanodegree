# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 15:23:38 2018

@author: SimonThornewill
"""

# =============================================================================
# Import Libraries
# =============================================================================

import argparse
import torch

from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


# =============================================================================
# Define Functions
# =============================================================================

def arg_parsing():
    """
    Parses keyword arguments from the command line to customise the training
    process.
    
    returns: Namespace with various strings, ints, floats and bools.
    """
    # Create parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    # Add architecture selection to parser
    parser.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture from torchvision.models as str')
    
    # Add checkpoint directory to parser
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Define save directory for checkpoints as str. If not specified then model will be lost.')
    
    # Add hyperparameter tuning to parser
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Define gradient descent learning rate as float')
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Hidden units for DNN classifier as int')
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training as int')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')
    
    # Parse args
    args = parser.parse_args()
    
    return args


def train_transformation(train_dir):
    """
    Does training transformations on a dataset
    
    train_dir: string, directory for training data
    
    returns: training set using torchvision.datasets
    """

    # Define transformation
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    # Load the Data
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    
    return train_data


def test_transformation(test_dir):
    """
    Does test/validation transformations on a dataset
    
    test_dir: string, directory for training data
    
    returns: testing set using torchvision.datasets 
    """
    
    # Define transformation
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    # Load the Data
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    return test_data
    

def data_loader(data, train=True):
    """
    Creates a dataloader from dataset imported with eitner the 
    test_transformation() or the train_transformation() functions.
    
    data: Dataset created using the functions mentioned above.
    
    train: bool, whether the transformation should be for a train or 
    test/validation set.
        
    returns: Dataloader using torch.utils.data.DataLoader.
    
    """
    
    if train: loader = torch.utils.data.DataLoader(data, 
                                                   batch_size=64, 
                                                   shuffle=True)
    else: loader = torch.utils.data.DataLoader(data, 
                                               batch_size=32)
    
    return loader


def check_gpu(gpu_arg):
    """
    Function decides whether to use CUDA with GPU or use CPU instead.
    
    gpu_arg
    
    returns: string,       
    """
    
    # If gpu_arg is false then simply return the cpu device
    if not gpu_arg: return torch.device("cpu")
    
    # If gpu_arg then make sure to check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Print result
    if device == "cpu": print("CUDA was not found on device, using CPU instead.")

    return device


def load_primary_model(architecture="vgg16"):
    """
    Downloads model (primary) from torchvision. Models imported from a 
    checkpoint are secondary models.
    
    architecture: string, Defines what model will be imported from torchvision
    
    returns: torchvision.models model
    """
    
    # Load Defaults if none specified
    if type(architecture) == type(None): 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network architecture not specificed, assuming vgg16.")
    else: 
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 
    
    return model


def create_classifier(model, hidden_units):
    """
    Creates a classifier with the corect number of input layers from the frozen
    output of the downloaded model using load_primary_model(). The number of
    input layers can be changed using kwargs in the command line.
    
    model: pretrained torchvision model
    
    returns: classifier, to be attached at the end of the pretrained model
    """

    # Check that hidden layers has been input
    if type(hidden_units) == type(None): 
        hidden_units = 4096
        print("Hidden layers not specified, assuming 4096 units")
    
    # Find Input Layers
    input_features = model.classifier[0].in_features
    
    # Define Classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.4)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    return classifier


def validation(model, testloader, criterion, device):
    """
    Does validation using a model in the process of training against a 
    testloader and some criteria. 
        
    returns: loss and accuracy
    """
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy


def train_network(Model, Trainloader, Testloader, Device, 
                  Criterion, Optimizer, Epochs, Print_every, Steps):

    """
    Train neural network.
    
    Model: Downloaded model
    
    Trainloader, Testloader: Data Loaders
    
    Criterion: Loss function for training
    
    Device: Whether to train using GPU or CPU.
    
    Optimizer: Gradient Descent Method.
    
    Epochs: Number of times to train on the training set.
    
    Print_every: Frequency of prints in steps.
        
    Steps: How many steps were taken.
    """
    
    # Check Model Kwarg
    if type(Epochs) == type(None):
        Epochs = 3
        print("Number of Epochs rate not specificed, assuming 3 epochs.")    
 
    print("Initiating training squence...")

    # Train Model
    for e in range(Epochs):
        running_loss = 0
        Model.train() # Technically not necessary, setting this for good measure
        
        for ii, (inputs, labels) in enumerate(Trainloader):
            Steps += 1
            
            inputs, labels = inputs.to(Device), labels.to(Device)
            
            Optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = Model.forward(inputs)
            loss = Criterion(outputs, labels)
            loss.backward()
            Optimizer.step()
            
            running_loss += loss.item()
            
            if Steps % Print_every == 0:
                Model.eval()
    
                with torch.no_grad():
                    valid_loss, accuracy = validation(Model, Testloader, 
                                                      Criterion, Device)
                
                print("Epoch: {}/{}.. ".format(e+1, Epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/Print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(Testloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(Testloader)))
                running_loss = 0
                Model.train()
    
    return Model

def validate_model(Model, Testloader, Device):
    """
    Checks the model, trained on train/test sets on a validation set
    
    Model: Trained model.
    Testloader: Data loader.
    Device: GPU or CPU
    
    Prints the accuracy of the network on test images
    """
    
    # Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in Testloader:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def create_checkpoint(Model, Save_Dir, Train_data):
    """
    Saves trained model to a specified save directory with the classifications
    of the training data.
    
    Model: Trained model to be saved
    
    Save_Dir: str, directory to be saved
    
    Train_Data: Traiing data used for forward/backpropogation on the model.
    """
    
    # Save model at checkpoint
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(Save_Dir):
            # Create `class_to_idx` attribute in model
            Model.class_to_idx = Train_data.class_to_idx
            
            # Create checkpoint dictionary
            checkpoint = {'architecture': Model.name,
                          'classifier': Model.classifier,
                          'class_to_idx': Model.class_to_idx,
                          'state_dict': Model.state_dict()}
            
            # Save checkpoint
            torch.save(checkpoint, Save_Dir+"/checkpoint.pth")

        else: print("Directory not found, model will not be saved.")


# =============================================================================
# Main Function
# =============================================================================

def main():
    """
    Executing relevant functions
    """
    
    # Get Keyword Args for Training
    args = arg_parsing()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = test_transformation(train_dir)
    valid_data = train_transformation(valid_dir)
    test_data = train_transformation(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    # Load Model
    model = load_primary_model(architecture=args.arch)
    
    # Build Classifier
    model.classifier = create_classifier(model, 
                                         hidden_units=args.hidden_units)
     
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);
    
    # Send model to device
    model.to(device);
    
    # Check for learnrate args
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate not specificed, assuming learning rate of 0.001.")
    else: learning_rate = args.learning_rate
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Define deep learning method
    print_every = 40 # Prints every 40 images, 64 images in a batch
    steps = 0
    

    
    # Train the classifier layers using backpropogation
    trained_model = train_network(model, trainloader, validloader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps)
    
    print("Training Complete")
    
    # Quickly Validate the model
    validate_model(trained_model, testloader, device)
    
    # Save the model
    create_checkpoint(trained_model, args.save_dir, train_data)


# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()
    