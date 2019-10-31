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
import numpy as np
from skimage.transform import resize


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


def save_history(history, trainer, param_name=None):

    if param_name:
        param = trainer.params.dict[param_name]
        filename = trainer.experiment_dir / f'history_{param_name}_{param}.pickle'
    else:
        filename = trainer.experiment_dir / 'history.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(history.history, f)


def load_history(filename):
    with open(filename, "rb") as f:
        history = pickle.load(f)
    return history


def plot_metric(metric, dir_path):
    for path in dir_path.glob('hist*'):
        history = load_history(path)
        param = get_param(path)
        plt.plot(history[metric], label=f'{param}')
    plt.legend()
    plt.title(metric)


def get_param(path):
    filename = str(path)
    return filename.split('_')[-1][:-7]


def random_transforms(img_arr, n_trans=3):

    all_transforms = [
        lambda x: x,
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.rot90(x, 1),
        lambda x: np.rot90(x, 2),
        lambda x: np.rot90(x, 3),
    ]

    for i in range(n_trans):
        idx = np.random.randint(0, len(all_transforms))
        transform = all_transforms[idx]
        img_arr = transform(img_arr)

    return img_arr


def random_crop(img_arr, crop_size=64):
    # Note: image_data_format is 'channel_last'
    assert img_arr.shape[2] == 1
    height, width = img_arr.shape[0], img_arr.shape[1]
    dy, dx = crop_size, crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img_arr[y:(y + dy), x:(x + dx), :]


def random_crop_batch(batch, crop_size=64):
    n, _, _, c = batch.shape
    batch_crop = np.zeros((n, crop_size, crop_size, c))
    for i in range(n):
        batch_crop[i] = random_crop(batch[i], crop_size=crop_size)
    return batch_crop


# def search_lr(learning_rates=(.001, .0005, .0001, .00005)):
#     params = Params('experiments/learning_rates/params.json')
#     for lr in learning_rates:
#         print(f'lr={lr}')
#         params.learning_rate = lr
#         trainer = Trainer(params=params)
#         history = trainer.train()
#         save_history(history, trainer, param_name='lr')


def resize_batch(batch, target_size):
    n, _, _, c = batch.shape
    batch_resize = np.zeros((n, target_size, target_size, c))
    for i in range(n):
        batch_resize[i] = resize(batch[i], output_shape=(target_size, target_size, 1))
    return batch_resize





if __name__ == '__main__':
    dir_path = Path('experiments/learning_rates/')
    plot_metric('loss', dir_path)
    plt.show()
