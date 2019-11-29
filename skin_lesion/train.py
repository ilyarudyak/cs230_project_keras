import tensorflow as tf
from pathlib import Path
import utils
from data.data_gen import SkinLesionDataGen
from model.full_unet import FullUnet
from model.bigger_leaky_unet import BiggerLeakyUnet


class Trainer:

    def __init__(self,
                 params=None,
                 experiment_dir=Path('experiments/full_unet'),
                 NetClass=None,

                 swap_gen=True,
                 train_gen=None,
                 val_gen=None,
                 ):

        # parameters
        if params:
            self.params = params
        else:
            self.params = utils.Params(experiment_dir / 'params.json')

        # net and model
        self.net = NetClass(params=self.params)
        self.model = self.net.get_model()

        # directories and files
        self.data_dir = Path.home() / 'data/isic_2018'
        self.experiment_dir = experiment_dir
        self.weight_file = self.experiment_dir / 'weights'

        # data generators
        if swap_gen:
            self.train_gen = train_gen
            self.val_gen = val_gen
        else:
            self.data_gen = SkinLesionDataGen(params=self.params)
            self.train_gen = self.data_gen.get_train_gen()
            self.val_gen = self.data_gen.get_val_gen()

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=self.params.learning_rate)

        # metrics and loss
        # self.metrics = ['accuracy', pixel_diff]
        # self.loss = self.params.loss
        self.metrics = [utils.jaccard_coef]
        self.loss = utils.jaccard_coef_loss
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        # callbacks
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(str(self.weight_file),
                                               save_weights_only=True,
                                               monitor='val_loss',
                                               save_best_only=True,
                                               verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.75,
                                                 patience=5,
                                                 cooldown=3,
                                                 min_lr=1e-6,
                                                 verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=1e-3,
                                             patience=20,
                                             mode='min',
                                             verbose=1)
        ]

    def train(self, load_weights=False):

        if load_weights:
            self.model.load_weights(self.weight_file)

        history = self.model.fit_generator(generator=self.train_gen,
                                           steps_per_epoch=self.params.num_train//self.params.batch_size,
                                           epochs=self.params.epochs,
                                           validation_data=self.val_gen,
                                           validation_steps=self.params.num_val//self.params.batch_size,
                                           callbacks=self.callbacks)
        return history

    def predict(self, weight_file):
        pass


if __name__ == '__main__':
    trainer = Trainer(experiment_dir=Path('experiments/bigger_leaky_unet'),
                      NetClass=BiggerLeakyUnet)
    history = trainer.train()
    utils.save_history(history, trainer)
