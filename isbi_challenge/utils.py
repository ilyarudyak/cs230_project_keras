import json
import logging
import os
import shutil
import keras.backend as K


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


def pixel_difference(y_true, y_pred, params):
    """
    Custom metrics for comparison of images
    pixel by pixel.
    """
    input_shape = params.input_shape
    batch_size = params.batch_size

    cof = 100 / (input_shape[0] * input_shape[1] * batch_size)
    return cof * K.sum(K.abs(y_true - y_pred))
