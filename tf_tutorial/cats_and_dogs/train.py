from pathlib import Path
import utils
from model import ConvNet
from data_prep import get_generators
import numpy as np


class Trainer:

    def __init__(self,
                 experiment_dir=Path('experiments/base_model'),
                 ):
        self.params = utils.Params(experiment_dir / 'params.json')

        # data generators
        self.train_data_gen, self.val_data_gen = get_generators()

        # model
        net = ConvNet()
        self.model = net.get_model()

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self):
        history = self.model.fit_generator(
            self.train_data_gen,
            steps_per_epoch=int(np.ceil(self.params.total_train / float(self.params.BATCH_SIZE))),
            epochs=self.params.EPOCHS,
            validation_data=self.val_data_gen,
            validation_steps=int(np.ceil(self.params.total_val / float(self.params.BATCH_SIZE)))
        )
        return history


if __name__ == '__main__':
    trainer = Trainer()
    history = trainer.train()
    utils.save_history(history, trainer)
