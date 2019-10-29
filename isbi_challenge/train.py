# from model.unet_v1 import *
from model.unet_v2 import Unet
from model.data_gen import *
# from utils import *
import utils
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import tifffile as tiff
import os
import keras.backend as K

import warnings

warnings.filterwarnings('ignore')


class Trainer:

    def __init__(self,
                 params,
                 experiment_dir=Path('experiments/learning_rates')
                 ):

        # parameters
        # self.params = Params(experiment_dir / 'params.json')
        self.params = params

        # model
        self.model = Unet(params=self.params).build_model()

        # directories and files
        self.data_dir = Path.home() / 'data/isbi2012'
        self.experiment_dir = experiment_dir
        # self.weight_file = self.experiment_dir / 'weights'
        self.pred_file = self.experiment_dir / 'test-volume-masks.tif'

        if not os.path.exists(str(self.experiment_dir)):
            os.makedirs(str(self.experiment_dir))

        # dataset
        self.dataset = ISBI2012(data_dir=self.data_dir)

        # optimizer
        self.optimizer = Adam(lr=self.params.learning_rate)

        # metrics and loss
        # self.metrics = ['accuracy', pixel_diff]
        # self.loss = self.params.loss
        self.metrics = [utils.dice_coef]
        self.loss = utils.bcdl_loss
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        # callbacks
        weight_file = str(self.experiment_dir / 'weights')
        self.callbacks = [
            TensorBoard(log_dir=str(self.experiment_dir),
                        update_freq='epoch'),
            ModelCheckpoint(weight_file + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                            save_weights_only=True,
                            monitor='val_loss',
                            save_best_only=True),
            EarlyStopping(monitor='val_loss',
                          patience=5,
                          verbose=1)
        ]

    def pixel_diff(self, y_true, y_pred):
        input_shape = self.params.input_shape
        batch_size = self.params.batch_size

        cof = 100 / (input_shape[0] * input_shape[1] * batch_size)
        return cof * K.sum(K.abs(y_true - y_pred))

    def train(self, load_weights=False):

        if load_weights:
            self.model.load_weights(self.weight_file)

        train_generator = self.dataset.generator('training', batch_size=self.params.batch_size)
        valid_generator = self.dataset.generator('validation', batch_size=self.params.batch_size_val)

        history = self.model.fit_generator(generator=train_generator,
                                           steps_per_epoch=self.params.steps_per_epoch,
                                           epochs=self.params.epochs,
                                           initial_epoch=self.params.start_epoch,
                                           validation_data=valid_generator,
                                           validation_steps=self.params.validation_steps,
                                           callbacks=self.callbacks)
        return history

    def predict(self, weight_file):
        self.model.load_weights(weight_file)
        self.dataset.load_data_test()
        test_masks = self.model.predict(self.dataset.image_data_test / 255.0,
                                        batch_size=self.params.batch_size_test)
        test_masks = test_masks.round()
        tiff.imsave(self.pred_file, test_masks)

    def predict_image(self, img_arr, weight_file):
        self.model.load_weights(weight_file)
        img_arr = np.expand_dims(img_arr / 255., axis=0)
        mask_pred = self.model.predict(img_arr).round()
        mask_pred = mask_pred.squeeze(axis=0)
        return mask_pred


if __name__ == '__main__':

    learning_rates = [.1, .01, .005, .001, .0005, .0001]
    params = utils.Params('experiments/learning_rates/params.json')
    for lr in learning_rates:
        print(f'lr={lr}')
        params.learning_rate = lr
        trainer = Trainer(params=params)
        history = trainer.train()
        utils.save_history(history, trainer)
