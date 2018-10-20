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
import json
# from train import load_data

parser = argparse.ArgumentParser(description='Pridect test data by loading trained model')
parser.add_argument('img_path', action="store", default='flowers/test/1/image_06743.jpg', help='Your image data path ex:flowers/test/1/image_06743.jpg')
parser.add_argument('checkpoint', action="store", default='checkpoint.pth', help='Your model save as .pth ex:checkpoint.pth')
parser.add_argument('--top_k', action='store', type=int, default=5, help='The top K classes along with associated probabilities you want to print')
parser.add_argument('--category_names', action='store', default='cat_to_name.json', help='Load in a mapping from category label to category name.')
parser.add_argument('--gpu', action='store_true', default=False, help='Use the GPU to calculate the predictions')

ARGS = vars(parser.parse_args())

def load_checkpoint(filepath, device):
    # Load all tensors onto the CPU, using a function
    try:
        if not device:
            # This is from torch document "torch.load" https://pytorch.org/docs/stable/torch.html
            checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        if device:
            checkpoint = torch.load(filepath)
    except:
        return False, False

    # you can re-build the exact model using just the parameters you had saved
    # in your checkpoint without directly referencing the model object you had used during the training.
    model = getattr(models, checkpoint['model_arch'])(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']

    model.epochs = checkpoint['epochs']
    model.learning_rate = checkpoint['learning_rate']
    model.arch = checkpoint['model_arch']

    return model, checkpoint['class_to_idx']

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    atransforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    image = atransforms(image)
    return image

def predict(image_path, device, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    try:
        img_ori = Image.open(image_path)
    except IOError:
        return False, False
    img = process_image(img_ori)
    img = Variable(img.unsqueeze(0))

    with torch.no_grad():
        if device:
            model.cuda()
            img = img.cuda()
        output = model.forward(img.float())

    ps = torch.exp(output)
    probs, classes = ps.topk(topk)
#    help by https://discuss.pytorch.org/t/convert-to-numpy-cuda-variable/499/4
    if device:
        probs = probs.data[0].cpu().numpy()
        classes = classes.data[0].cpu().numpy()
    if not device:
        probs = probs.data[0].numpy()
        classes = classes.data[0].numpy()
    return [x for x in probs], [y for y in classes]

def main():

    model, class_to_idx = load_checkpoint(ARGS['checkpoint'], ARGS['gpu'])
    if not model:
        print("Can't not load {}".format(ARGS['checkpoint']))
        exit(1)
    print("1/2 Load checkpoint - Done")

    with open(ARGS['category_names'], 'r') as f:
        cat_to_name = json.load(f)

    idx_to_classes = {v :k for k, v in class_to_idx.items()}
    probs, classes = predict(ARGS['img_path'], ARGS['gpu'], model, ARGS['top_k'])
    if not probs:
        print("The {} cannot be found, or the image cannot be opened and identified.".format(ARGS['img_path']))
        exit(1)
    print("Proability top {} = {}".format(ARGS['top_k'], probs))
    print("Classes index top {} = {}".format(ARGS['top_k'], [idx_to_classes[x] for x in classes]))
    # In slack  #office-hours Rami Ejleh asked and answer by Abdel Affo,
    # help my figure how to use cat_to_name
    print("Classes name top {} = {}\n".format(ARGS['top_k'], [cat_to_name[idx_to_classes[x]] for x in classes]))
    print("{} is {} in proability = {:.4f}\n".format( ARGS['img_path'], cat_to_name[idx_to_classes[classes[0]]], probs[0]))

    print("2/2 Pridect - Done")

if __name__ == '__main__':
    main()
