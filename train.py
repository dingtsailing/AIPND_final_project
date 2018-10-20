import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import time
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
import numpy as np
import argparse
from glob import glob

parser = argparse.ArgumentParser(description='Train models')
parser.add_argument('data_directory', action="store", default='flowers/', help='Image data path')
parser.add_argument('--save_dir', action="store", default='checkpoint.pth', help='The path you want to save the model')
parser.add_argument('--arch', action="store", default='vgg16', help='Choose a pretrained model in vgg13 or vgg16 or densenet121 (from torchvision.models)')
parser.add_argument('--learning_rate', action="store", type=float, default=0.001, help='Set learning_rate ex:0.001')
parser.add_argument('--hidden_units', action="store", type=int, default=2048, help='Set hidden layer units ex:2048 (if you choose densenet121 should set 512)')
parser.add_argument('--epochs', action="store", type=int, default=3, help='Set training epochs ex:3')
parser.add_argument('--gpu', action="store_true", default=False, help='Training the model on a GPU')

# help by https://stackoverflow.com/questions/7427101/simple-argparse-example-wanted-1-argument-3-results
ARGS = vars(parser.parse_args())

def load_data(train_dir, test_dir, valid_dir):
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               [0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])])
    # The resizing is important here because if we do not resize the image and
    # directly crop it, we might lose important information about the data image.
    # Resizing reduces the size of a image while still holding full information
    # of the image unlike a crop which blindly extracts one part of the image.
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              [0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])
    data_transforms = {'train':train_transforms, 'test':test_transforms}

    # Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)
    image_datasets = {'train':train_image_datasets, 'test':test_image_datasets,
                      'valid':valid_image_datasets}

    # Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=64)
    validloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64)
    dataloaders = {'train':trainloaders, 'test':testloaders,
                   'valid':validloaders}
    return data_transforms, image_datasets, dataloaders

def load_set_model(arch, hidden_units):
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        # Obtaining log-probabilities in a neural network is easily achieved by adding
        # a LogSoftmax layer in the last layer of the network.
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1024, hidden_units)),
                                                ('relu1', nn.ReLU()),
                                                ('fc2', nn.Linear(hidden_units, 128)),
                                                ('relu2', nn.ReLU()),
                                                ('fc3', nn.Linear(128, 102)),
                                                ('output', nn.LogSoftmax(dim=1))
                                               ]))
    else:
        if arch == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif arch == 'vgg13':
            model = models.vgg13(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        # Obtaining log-probabilities in a neural network is easily achieved by adding
        # a LogSoftmax layer in the last layer of the network.
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 4096)),
                                                ('relu1', nn.ReLU()),
                                                ('fc2', nn.Linear(4096, hidden_units)),
                                                ('relu2', nn.ReLU()),
                                                ('fc3', nn.Linear(hidden_units, 102)),
                                                ('output', nn.LogSoftmax(dim=1))
                                               ]))
    model.classifier = classifier
    return model, classifier

def do_deep_learning(model, trainloader, validloader, epochs, print_every,
                     criterion, optimizer, device):
    epochs = epochs
    print_every = print_every
    steps = 0

    if device and torch.cuda.is_available():
        model.to('cuda')
    if not device:
        model.to('cpu')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            # set training mode
            model.train()

            inputs, labels = Variable(inputs), Variable(labels)
            if device:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                correct = 0
                total = 0
                # testing/validating your model
                # Many layers like Dropout behave differently during training and
                # validation, so this is a very important step and should not be missed out.
                model.eval()
                for ii, (valid_inputs, valid_labels) in enumerate(validloader):
                    valid_inputs, valid_labels = Variable(valid_inputs), Variable(valid_labels)
                    if device:
                        valid_inputs, valid_labels = valid_inputs.to('cuda'),\
                        valid_labels.to('cuda')
                    outputs = model.forward(valid_inputs)
                    validation_loss += criterion(outputs, valid_labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += valid_labels.size(0)
                    correct += (predicted == valid_labels).sum().item()

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(validation_loss/len(validloader)),
                      'Accuracy: %d %%' %  (100 * correct / total))
                running_loss = 0
    return model

def check_accuracy_on_test(testloader, model, device):
    if device:
        model.to('cuda')
    if not device:
        model.to('cpu')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = Variable(images), Variable(labels)
            if device:
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total

def save_checkpoint(name, model, optimizer, classifier):
    checkpoint = {'optimizer':optimizer.state_dict(),
                  'class_to_idx':model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'classifier': classifier,
                  'model_arch': ARGS['arch'],
                  'learning_rate': ARGS['learning_rate'],
                  'epochs': ARGS['epochs']}

    torch.save(checkpoint, name)

def main():
    # help by https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
    folder = glob(ARGS['data_directory']+'/*/')
    folder = [x.split('/')[1] for x in folder]
    if 'train' not in folder or 'test' not in folder or 'valid' not in folder:
        print("You must have all of these three 'train, test and vaild folder'\
         in your data_directory.")
        # help by https://help.semmle.com/wiki/pages/viewpage.action?pageId=5933809
        exit(1)

    train_dir = ARGS['data_directory'] + 'train'
    test_dir = ARGS['data_directory'] + 'test'
    valid_dir = ARGS['data_directory'] + 'valid'

    data_transforms, image_datasets, dataloaders = load_data(train_dir, test_dir, valid_dir)
    print("1/4 Load data - Done")
    model, classifier = load_set_model(ARGS['arch'], ARGS['hidden_units'])
    print("2/4 Load pretrain model and set classifier -  Done")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=ARGS['learning_rate'])

    print_every = 25
    model = do_deep_learning(model, dataloaders['train'], dataloaders['valid'],
                             ARGS['epochs'], print_every, criterion, optimizer, ARGS['gpu'])
    print("3/4 Train model - Done")

    correct, total = check_accuracy_on_test(dataloaders['test'], model, ARGS['gpu'])
    print('Accuracy of the network: %d %%' %  (100 * correct / total))

    model.class_to_idx = image_datasets['train'].class_to_idx
    save_checkpoint(ARGS['save_dir'], model, optimizer, classifier)
    print("4/4 Save checkpoint - Done")


if __name__ == '__main__':
    main()
