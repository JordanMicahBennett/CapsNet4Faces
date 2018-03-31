#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Train the model.

Usage:
  train.py [--ckpt=<ckpt>]

Options:
  -h --help     Show this help.
  <ckpt>        Path to the checkpoints to restore
"""

from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from PIL import Image, ImageEnhance
from docopt import docopt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import random
import pickle
import os

from model import FaceRec
from data_handler import get_face_data


def train(ckpt=None):
    """
        Test the model
        **input: **
            *ckpt: (String) [Optional] Path to the ckpt file to restore
    """

    X_train, _, X_test, y_test = get_face_data()

    import pdb ; pdb.set_trace()
    X_test = X_test / 255

    # Utils method to print the current progression
    def plot_progression(b, cost, acc, label):
        print("[%s] Batch ID = %s, loss = %s, acc = %s" % (label, b, cost, acc))

    # Init model
    model = FaceRec("FaceRec", output_folder='/tmp')

    if ckpt is None:
        model.init()
    else:
        model.load(ckpt)

    print("Loaded model. Beginning to test the entire test dataset")
    print("Size of error dataset: %s/%s" %(X_test.shape[0], (X_test.shape[0]+X_train.shape[0])))

    loss, acc, _ = model.evaluate_dataset(X_test, y_test)
    plot_progression(0, loss, acc, "Total Test Validation")


if __name__ == '__main__':
    arguments = docopt(__doc__)
    train(arguments["--ckpt"])
