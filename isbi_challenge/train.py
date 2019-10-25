import keras
import sys
import os
from glob import glob
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from callbacks import ImageHistory
from pathlib import Path
import time

from model.unet_v1 import *
from model.data_gen import *
from model.callbacks import *
from utils import *
from keras.optimizers import Adam

import warnings

warnings.filterwarnings('ignore')
import tensorflow.python.util.deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


class Trainer:

    def __init__(self,
                 experiment_dir=Path('experiments/initial_model')):

        # parameters
        self.params = Params(experiment_dir / 'params.json')

        #model
        self.model = Unet(params=self.params).build_model()

        # directories and files
        self.data_dir = Path.home() / 'data/isbi2012'
        self.weights_dir = Path.home() / 'weights'
        self.weight_file = self.weights_dir / f'weights_{int(time.time())}.hdf5'
        self.experiment_dir = experiment_dir

        if not os.path.exists(str(self.weights_dir)):
            os.makedirs(str(self.weights_dir))

        if not os.path.exists(str(self.experiment_dir)):
            os.makedirs(str(self.experiment_dir))

        # dataset
        self.dataset = ISBI2012(data_dir=self.data_dir)

        # optimizer
        self.optimizer = Adam(lr=self.params.learning_rate)

        # metrics
        self.metrics = ['accuracy', pixel_difference]

    def train(self):
        pass


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
