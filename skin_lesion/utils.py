import json
import tensorflow.keras.backend as K
import pickle
import matplotlib.pyplot as plt


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


# Define loss function
def jaccard_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def jaccard_coef_loss(y_true, y_pred):
    j = -jaccard_coef(y_true, y_pred)
    return j


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


def plot_metric_paths(metric, paths):
    for path in paths:
        history = load_history(path)
        param = get_param(path)
        plt.plot(history[metric], label=f'{param:.1e}')
    plt.legend()
    plt.title(metric)


def get_param(path):
    filename = str(path)
    param = float(filename.split('_')[-1][:-7])
    return param



