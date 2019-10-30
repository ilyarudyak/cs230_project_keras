import json
import logging
import os
import shutil
import keras.backend as K
import tensorflow as tf
from keras import losses
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


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
    # print(y_true_flat, y_true_flat_flip)
    y_pred_flat = K.flatten(y_pred)
    y_pred_flat_flip = 1 - tf.round(y_pred_flat)
    # print(y_pred_flat, y_pred_flat_flip)

    # print((2 * K.sum(y_true_flat * y_pred_flat) + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth))

    intersection = K.sum(y_true_flat_flip * y_pred_flat_flip)
    sum_smooth = K.sum(y_true_flat_flip) + K.sum(y_pred_flat_flip) + smooth
    return (2.0 * intersection + smooth) / sum_smooth


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bcdl_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def plot_masks(img_arr, masks):
    fig, ax = plt.subplots(1, len(masks)+1, figsize=(15, 15))
    [axi.set_xticks([]) for axi in ax.ravel()]
    [axi.set_yticks([]) for axi in ax.ravel()]
    ax[0].imshow(img_arr.reshape(512, 512), cmap='gray')
    ax[0].title.set_text('image')
    for i in range(len(masks)):
        ax[i+1].imshow(masks[i].reshape(512, 512), cmap='gray')
        ax[i+1].title.set_text(f'mask_{i+1}')


def save_history(history, trainer, is_lr=False):
    lr = trainer.params.learning_rate
    if is_lr:
        filename = trainer.experiment_dir / f'history_lr_{lr}.pickle'
    else:
        filename = trainer.experiment_dir / 'history.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(history.history, f)


def load_history(filename):
    with open(filename, "rb") as f:
        history = pickle.load(f)
    return history


def plot_metric_lr(metric, dir_path):
    for path in dir_path.glob('hist*'):
        history = load_history(path)
        lr = get_lt(path)
        plt.plot(history[metric], label=f'{lr}')
    plt.legend()
    plt.title(metric)


def get_lt(path):
    filename = str(path)
    return filename.split('_')[-1]


if __name__ == '__main__':
    dir_path = Path('experiments/learning_rates/')
    plot_metric_lr('loss', dir_path)
    plt.show()
