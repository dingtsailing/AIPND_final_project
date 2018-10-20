# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

How to use:

For train.py:
    Basic usage: python train.py data_directory
    Options:
    Set diectory to save checkpoint: python train.py data_dir --save_dir save_directory
    Choose architecture: python train.py data_dir --arch "vgg13"
                         only can choose vgg13 or vgg16 or densenet121 (from torchvision.models)
    Set hyperparameters: python train.py data_dir -- learning_rate 0.001 --hidden_units 2048 --epochs 3
                         if you architecture choose densenet121 should set 512
    Use GPU for training: python train.py data_dir --gpu

    Example: python train.py flowers/ --save_dir checkpoint.pth --arch vgg16 --learning_rate 0.001 --hidden_units 2048 --epochs 3 --gpu => can get 84% accurary



For predict.py:
    Basic usage: python predict.py /path/to/image checkpoint
    Option:
    Return top K most likely classes: python predict.py /path/to/image checkpoint -- top_k 3
    Use a mapping of categories to real names: python predict.py /path/to/image checkpoint --category_names cat_to_name.json
    Use GPU for inference: python predict.py /path/to/image checkpoint --gpu

    Example: python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu

