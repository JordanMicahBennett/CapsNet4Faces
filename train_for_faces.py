#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Train the model.

Usage:
  train.py [<output>] [--ckpt=<ckpt>] [--batch_size=<batch_size>]

Options:
  -h --help     Show this help.
  <batch_size>  Batch size to train on
  <output>      Ouput folder. By default: ./outputs/
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


BATCH_SIZE = 10
EARLY_STOPPING_COUNT = 10

def train(batch_size=None, ckpt=None, output=None):
    """
        Train the model
        **input: **
            *dataset: (String) Dataset folder to used
            *ckpt: (String) [Optional] Path to the ckpt file to restore
            *output: (String) [Optional] Path to the output folder to used. ./outputs/ by default
    """
    if not batch_size:
        batch_size = BATCH_SIZE
    batch_size = int(batch_size)
    print("Batch size: %s" %(batch_size))

    X_train, y_train, X_test, y_test = get_face_data()

    X_train = X_train / 255
    X_test = X_test / 255

    ## do a quick check on the number of labels expected by model
    ## and number of models passed in
    nb_labels_dataset = max(len(set(y_train)),
                            len(set(y_test)))
    if nb_labels_dataset != FaceRec.NB_LABELS:
        print("Number of labels is mismatched.. can't train this way")
        return

    train_datagen = ImageDataGenerator()
    train_datagen_augmented = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
    inference_datagen = ImageDataGenerator()
    train_datagen.fit(X_train)
    train_datagen_augmented.fit(X_train)
    inference_datagen.fit(X_test)

    # Utils method to print the current progression
    def plot_progression(b, cost, acc, label):
        print("[%s] Batch ID = %s, loss = %s, acc = %s" % (label, b, cost, acc))

    # Init model
    model = FaceRec("FaceRec", output_folder=output)

    if ckpt is None:
        model.init()
    else:
        model.load(ckpt)

    # Training pipeline
    b = 0
    best_validation_loss = float('inf')
    augmented_factor = 0.99
    decrease_factor = 0.80
    train_batches = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    augmented_train_batches = train_datagen_augmented.flow(X_train, y_train, batch_size=batch_size)
    valid_batch = inference_datagen.flow(X_test, y_test, batch_size=batch_size)

    while True:
        next_batch = next(
            augmented_train_batches if random.uniform(0, 1) < augmented_factor else train_batches)

        ### Training
        x_batch, y_batch = next_batch
        cost, acc = model.optimize(x_batch, y_batch)

        ### Validation --> with test data
        # Retrieve the cost and acc on this validation batch and save it in tensorboard
        x_batch, y_batch = next(valid_batch, None)
        cost_val, acc_val = model.evaluate(x_batch, y_batch, tb_test_save=True)
        count = EARLY_STOPPING_COUNT

        # Plot the last results
        if b % 10 == 0:
            plot_progression(b, cost, acc, "Train")
            plot_progression(b, cost_val, acc_val, "Validation")

            # Early stopping logic; if there is EARLY_STOPPING_COUNT
            # worth of consecutive accuracies of 100%, we stop
            if acc == 1:
                count -= 1
            else:
                count = EARLY_STOPPING_COUNT

            if not count:
                print("model has hit 100% accuracy and met early stopping criteria")
                model.save()
                break

        # every 100 batch sizes, we check if the model should be saved based
        # on if the model's loss on 80% of the test dataset
        if b % 100 == 0:
            # We decide whether to checkpoint based on 30% of the test dataset
            # this is just to speed up computation
            _, save_x_test, _, save_y_test = train_test_split(X_test, y_test,
                                                              test_size=0.8,
                                                              random_state=b)
            loss, acc, _ = model.evaluate_dataset(save_x_test, save_y_test)
            print("Current loss: %s Best loss: %s" % (loss, best_validation_loss))
            if loss < best_validation_loss:
                best_validation_loss = loss
                model.save()
            # as we get better result we do less augmentation
            augmented_factor = augmented_factor * decrease_factor
            print("Augmented Factor = %s" % augmented_factor)

        b += 1

if __name__ == '__main__':
    arguments = docopt(__doc__)
    train(arguments["--batch_size"], arguments["--ckpt"], arguments["<output>"])
