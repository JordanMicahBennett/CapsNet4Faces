#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import pickle
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np
import random

def downsample_image(img):
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img = img.resize((32,32), Image.ANTIALIAS)
    return np.array(img)

def get_face_data():
    people = fetch_lfw_people(color=True, min_faces_per_person=25)
    X_faces = people.images
    Y_faces = people.target

    X_faces = np.array([downsample_image(ab) for ab in X_faces])
    X_train, X_test, y_train, y_test = train_test_split(X_faces, Y_faces,
                                                        test_size=0.2,
                                                        random_state=13)
    return X_train, y_train, X_test, y_test
