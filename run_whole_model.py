import argparse
import json
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.image import imread

import datasplit
from train import train_model
from predict import run_example

# Hyperparameters
parser = argparse.ArgumentParser(description='NN')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--dir', type=str,  default='data/', help='Directory of dataset')
parser.add_argument('--batch-size', type=int, default=16, metavar='SIZE', help='Batch size')
parser.add_argument('--epochs', type=int, default=20, metavar='NUMBER', help='Number of epochs')
parser.add_argument('--train-size', type=float, default=0.8, help='Training/Test division')
parser.add_argument('--img-height', type=int, default=128, help='Height which images will be resized')
parser.add_argument('--img-width', type=int, default=128, help='Width which images will be resized')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Network learning rate')
parser.add_argument('--test-dir', type=str, default='data/', help='Name of image to test')
parser.add_argument('--model-name', type=str, default='binary_model', help='Name of Model')
args = parser.parse_args()


HEIGTH=args.img_height
WIDTH=args.img_width
BATCH_SIZE=args.batch_size
EPOCHS=args.epochs
dir = args.dir
l_r=args.learning_rate
filedir=args.test_dir
model_name=args.model_name
validation_dir=args.test_dir

#Split into 2 folders: train and validation and randomly copy the data to each
datasplit.split(args.dir, args.train_size, args.seed)

train_dir = dir + 'train/'
validation_dir = dir + 'validation/'

def run(args, model_name, train_dir, filedir,
        HEIGTH, WIDTH, BATCH_SIZE, EPOCHS, l_r,
        validation_dir,
        train=False, validation=False, eval=True):
    if train:
        train_model(model_name, train_dir,
            HEIGTH, WIDTH, BATCH_SIZE, EPOCHS,
            validation_dir, train=True)
    if validation:
        train_model(model_name, train_dir,
            HEIGTH, WIDTH, BATCH_SIZE, EPOCHS,
            validation_dir, train=False)

    model = load_model(model_name+'.h5')
    if eval:
        for filename in os.listdir(filedir):
            run_example(model, filedir+filename,HEIGTH,WIDTH)
#
# run(args, model_name, train_dir, filedir,
#         HEIGTH, WIDTH, BATCH_SIZE, EPOCHS, l_r,
#         validation_dir,
#         train=True, validation=False, eval=False)

train_model(model_name, train_dir,
    HEIGTH, WIDTH, BATCH_SIZE, EPOCHS,
    validation_dir, train=True)
