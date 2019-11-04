import json
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


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
        history_dict = pickle.load(f)
    return history_dict


def plot_history(filename):
    history_dict = load_history(filename)

    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')


def get_image_net_label(idx):
    _URL = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', _URL)
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    return imagenet_labels[idx]

