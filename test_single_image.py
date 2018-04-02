#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Get prediction for image

Usage:
  calculate_image.py [--ckpt=<ckpt>]

Options:
  -h --help     Show this help.
  <ckpt>        Path to the checkpoints to restore
"""

from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from PIL import Image, ImageEnhance
from PIL.PngImagePlugin import PngImageFile
from docopt import docopt
from sklearn.datasets import fetch_lfw_people
#import tensorflow as tf
import numpy as np
import random
import pickle
import os

from model import FaceRec
from data_handler import get_face_data

people = fetch_lfw_people(color=True, min_faces_per_person=10)

def downsample_image(img):
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img = img.resize((32,32), Image.ANTIALIAS)
    return np.array(img)

def get_image_prediction(ckpt=None):
    """
        Test the model
        **input: **
            *ckpt: (String) [Optional] Path to the ckpt file to restore
    """
    print("Loading model")
    model = FaceRec("FaceRec", output_folder='/tmp')

    if ckpt is None:
        model.init()
    else:
        model.load(ckpt)

    while True:
        image_name = input("\nType image name:  ")

        if image_name.lower() == "exit" or image_name.lower() == "e":
            break

        try:
            image = Image.open(image_name)
        except IOError:
            print("Invalid File. Please try again")
            continue

        if type(image) == PngImageFile:
            image = image.resize((32, 32), Image.ANTIALIAS).convert('RGB')
            image = np.array(image)
            image  = image / 255
        else:
            image = downsample_image(image)

        softmax = model.predict(np.array([image]))
        top_5 = softmax.argsort()[0][-5:][::-1]
        top_5_names = [(people.target_names[i], softmax[0][i]) for i in top_5]
        print("\nPredicted Names and softmax score: ")
        for name in top_5_names:
            print("%s: %s" %(name[0], name[1]))


if __name__ == '__main__':
    arguments = docopt(__doc__)
    get_image_prediction(arguments["--ckpt"])
