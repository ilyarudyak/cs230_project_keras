import tensorflow as tf
from utils import Params


class FullUnetVGG:
    """
    This model has the shape specified in the original paper,
    uses tf.keras.layers.UpSampling2D (not tf.keras.layers.Conv2DTranspose) and relu activation
    (not leaky relu). This model uses MaxPool, not BatchNorm.
    """

    def __init__(self,
                 params,
                 set_seed=False,
                 freeze=True):
        self.params = params
        self.input_shape = params.input_shape
        self.set_seed = set_seed
        self.kernel_initializer = params.kernel_initializer
        self.alpha = params.alpha

        self.base_model = tf.keras.applications.VGG16(weights='imagenet',
                                                      include_top=False)
        if freeze:
            self.freeze_layers()

        self.model = None

    def freeze_layers(self):
        for layer in self.base_model.layers:
            layer.trainable = False

    def get_model(self):
        if self.model is None:
            self._build_model()

        return self.model

    def _build_model(self):

        if self.set_seed:
            tf.random.set_seed(self.params.seed)

        x = self.base_model.output

        bottom_conv = tf.keras.layers.Conv2D(1024, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(x)
        bottom_acti = tf.keras.layers.LeakyReLU(alpha=self.alpha)(bottom_conv)
        bottom_conv = tf.keras.layers.Conv2D(1024, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(bottom_acti)
        bottom_acti = tf.keras.layers.LeakyReLU(alpha=self.alpha)(bottom_conv)

        # block5
        left_block5 = self.base_model.get_layer('block5_conv3').output
        right_up5 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(bottom_acti), left_block5],
                                                axis=3)
        right_conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_up5)
        right_acti5 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv5)
        right_conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_acti5)
        right_acti5 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv5)

        # block4
        left_block4 = self.base_model.get_layer('block4_conv3').output
        right_up4 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(right_acti5), left_block4],
                                                axis=3)
        right_conv4 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_up4)
        right_acti4 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv4)
        right_conv4 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_acti4)
        right_acti4 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv4)

        # block3
        left_block3 = self.base_model.get_layer('block3_conv3').output
        right_up3 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(right_acti4), left_block3],
                                                axis=3)
        right_conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_up3)
        right_acti3 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv3)
        right_conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_acti3)
        right_acti3 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv3)

        # block2
        left_block2 = self.base_model.get_layer('block2_conv2').output
        right_up2 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(right_acti3), left_block2],
                                                axis=3)
        right_conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_up2)
        right_acti2 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv2)
        right_conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_acti2)
        right_acti2 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv2)

        # block1
        left_block1 = self.base_model.get_layer('block1_conv2').output
        right_up1 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(right_acti2), left_block1],
                                                axis=3)
        right_conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_up1)
        right_acti1 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv1)
        right_conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_acti1)
        right_acti1 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv1)

        # sigmoid
        output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(right_acti1)
        self.model = tf.keras.models.Model(inputs=self.base_model.input, outputs=output)


if __name__ == '__main__':
    params = Params('../experiments/transf_learn_vgg_toy/params.json')
    net = FullUnetVGG(params=params)
    model = net.get_model()
    model.summary()
