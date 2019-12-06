import tensorflow as tf
from pathlib import Path
import utils
from data.data_gen import SkinLesionDataGen
from model.full_unet import FullUnet
from model.bigger_leaky_unet import BiggerLeakyUnet
from model.bigger_leaky_bn_unet import BiggerLeakyBNUnet


class Trainer:

    def __init__(self,
                 params=None,
                 experiment_dir=Path('experiments/bigger_leaky_unet'),
                 net_class=None,
                 set_seed=False,
                 is_toy=False
                 ):

        tf.keras.backend.clear_session()

        # parameters
        if params:
            self.params = params
        else:
            self.params = utils.Params(experiment_dir / 'params.json')

        # net and model
        self.net = net_class(params=self.params, set_seed=set_seed)
        self.model = self.net.get_model()

        # directories and files
        self.is_toy = is_toy
        if not is_toy:
            self.data_dir = Path.home() / 'data/isic_2018'
        else:
            self.data_dir = Path.home() / 'data/isic_2018/toy'

        self.experiment_dir = experiment_dir
        self.weight_file = self.experiment_dir / 'weights'

        # data generators
        self.data_gen = SkinLesionDataGen(params=self.params,
                                          data_dir=self.data_dir)
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
                                                 min_lr=1e-6,
                                                 verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=1e-3,
                                             patience=15,
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
        utils.save_history(history, self)
        return history

    def predict(self, weight_file):
        pass


class Tuner:

    def __init__(self,
                 params,
                 net_class,
                 experiment_dir,
                 is_toy=False,
                 set_seed=False):

        self.params = params
        self.net_class = net_class
        self.experiment_dir = experiment_dir
        self.set_seed = set_seed
        self.is_toy = is_toy
        self.trainer = None

    def tune_lr(self, rates=(1e-3, 1e-4, 1e-5, 1e-6)):
        for lr in rates:
            print(f'============== lr: {lr} ==============')

            self.params.learning_rate = lr
            self.trainer = Trainer(params=self.params,
                                   net_class=self.net_class,
                                   experiment_dir=self.experiment_dir,
                                   is_toy=self.is_toy,
                                   set_seed=self.set_seed)
            history = self.trainer.train()
            utils.save_history(history, self.trainer, param_name='learning_rate')

    def tune_batch_size(self, batch_sizes=(2, 4, 8, 16, 32)):
        for bs in batch_sizes:
            print(f'============== bs: {bs} ==============')
            self.params.batch_size = bs
            self.trainer = Trainer(params=self.params,
                                   net_class=self.net_class,
                                   experiment_dir=self.experiment_dir,
                                   is_toy=self.is_toy,
                                   set_seed=self.set_seed)
            history = self.trainer.train()
            utils.save_history(history, self.trainer, param_name='batch_size')

    def tune_leaky_relu(self, alphas=(0, .001, .01, .1, .2, .3), name_modifier=''):
        for alpha in alphas:
            print(f'============== alpha: {alpha} ==============')
            self.params.alpha = alpha
            self.trainer = Trainer(params=self.params,
                                   net_class=self.net_class,
                                   experiment_dir=self.experiment_dir,
                                   is_toy=self.is_toy,
                                   set_seed=self.set_seed)
            history = self.trainer.train()
            utils.save_history(history, self.trainer, param_name='alpha', name_modifier=name_modifier)

    def tune_kernel_initializer(self, kernel_initializers=('glorot_uniform', 'he_uniform', 'he_normal')):
        for kernel_initializer in kernel_initializers:
            print(f'============== kernel_initializer: {kernel_initializer} ==============')
            self.params.kernel_initializer = kernel_initializer
            self.trainer = Trainer(params=self.params,
                                   net_class=self.net_class,
                                   experiment_dir=self.experiment_dir,
                                   is_toy=self.is_toy,
                                   set_seed=self.set_seed)
            history = self.trainer.train()
            utils.save_history(history, self.trainer, param_name='kernel_initializer')

    def tune_image_shape(self, input_shapes=((512, 512, 3), (256, 256, 3))):
        for input_shape in input_shapes:
            print(f'============== input_shape: {input_shape} ==============')
            self.params.input_shape = input_shape
            self.trainer = Trainer(params=self.params,
                                   net_class=self.net_class,
                                   experiment_dir=self.experiment_dir,
                                   is_toy=self.is_toy,
                                   set_seed=self.set_seed)
            history = self.trainer.train()
            utils.save_history(history, self.trainer, param_name='input_shape')

    def tune_model_size(self, model_sizes=('regular', 'big')):

        net_classes = {'regular': FullUnet,
                       'big': BiggerLeakyUnet}

        for model_size in model_sizes:
            print(f'============== model_size: {model_size} ==============')
            self.params.model_size = model_size
            self.trainer = Trainer(params=self.params,
                                   net_class=net_classes[model_size],
                                   experiment_dir=self.experiment_dir,
                                   is_toy=self.is_toy,
                                   set_seed=self.set_seed)
            history = self.trainer.train()
            utils.save_history(history, self.trainer, param_name='model_size')

    def tune_batch_norm(self, normalizations=('no-batch-norm', 'batch-norm')):

        net_classes = {'no-batch-norm': BiggerLeakyUnet,
                       'batch-norm': BiggerLeakyBNUnet}

        for normalization in normalizations:
            print(f'============== normalization: {normalization} ==============')
            self.params.normalization = normalization
            if normalization == 'batch-norm':
                self.params.learning_rate = .1
            else:
                self.params.learning_rate = 1e-5
            self.trainer = Trainer(params=self.params,
                                   net_class=net_classes[normalization],
                                   experiment_dir=self.experiment_dir,
                                   is_toy=self.is_toy,
                                   set_seed=self.set_seed)
            history = self.trainer.train()
            utils.save_history(history, self.trainer, param_name='normalization')


if __name__ == '__main__':

    experiment_dir = Path('experiments/batch_norm_toy')
    params = utils.Params(experiment_dir / 'params.json')
    tuner = Tuner(params=params,
                  net_class=BiggerLeakyBNUnet,
                  experiment_dir=experiment_dir,
                  is_toy=True,
                  set_seed=True)
    tuner.tune_batch_norm(('batch-norm',))
