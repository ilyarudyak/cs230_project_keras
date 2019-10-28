import json
import logging
import os
import shutil
import keras.backend as K
import tensorflow as tf


class Params:
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_flat = K.flatten(y_true)
    y_true_flat_flip = 1 - tf.round(y_true_flat)
    print(y_true_flat, y_true_flat_flip)
    y_pred_flat = K.flatten(y_pred)
    y_pred_flat_flip = 1 - tf.round(y_pred_flat)
    print(y_pred_flat, y_pred_flat_flip)

    print((2 * K.sum(y_true_flat * y_pred_flat) + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth))

    intersection = K.sum(y_true_flat_flip * y_pred_flat_flip)
    sum_smooth = K.sum(y_true_flat_flip) + K.sum(y_pred_flat_flip) + smooth
    return (2.0 * intersection + smooth) / sum_smooth

