import tensorflow as tf
from utils import Params


class BiggerLeakyBNUnet:
    """
    ...
    """

    def __init__(self,
                 params,
                 set_seed=False):
        self.params = params
        self.input_shape = params.input_shape
        self.alpha = params.alpha
        self.kernel_initializer = params.kernel_initializer
        self.set_seed = set_seed

        self.model = None

    def get_model(self):
        if self.model is None:
            self._build_model()
        return self.model

    def _build_model(self):

        if self.set_seed:
            tf.random.set_seed(self.params.seed)

        inputs = tf.keras.layers.Input(self.input_shape)
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(inputs)
        norm1 = tf.keras.layers.BatchNormalization()(conv1)
        acti1 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(norm1)
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(acti1)
        norm1 = tf.keras.layers.BatchNormalization()(conv1)
        acti1 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(norm1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(acti1)

        conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(pool1)
        norm2 = tf.keras.layers.BatchNormalization()(conv2)
        acti2 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(norm2)
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(acti2)
        norm2 = tf.keras.layers.BatchNormalization()(conv2)
        acti2 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(norm2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(acti2)

        conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(pool2)
        norm3 = tf.keras.layers.BatchNormalization()(conv3)
        acti3 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(norm3)
        conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(acti3)
        norm3 = tf.keras.layers.BatchNormalization()(conv3)
        acti3 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(norm3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(acti3)

        conv4 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(pool3)
        norm4 = tf.keras.layers.BatchNormalization()(conv4)
        acti4 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(norm4)
        conv4 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(acti4)
        norm4 = tf.keras.layers.BatchNormalization()(conv4)
        acti4 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(norm4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(acti4)

        conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(pool4)
        norm5 = tf.keras.layers.BatchNormalization()(conv5)
        acti5 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(norm5)
        conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(acti5)
        norm5 = tf.keras.layers.BatchNormalization()(conv5)
        acti5 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(norm5)
        pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(acti5)

        conv6 = tf.keras.layers.Conv2D(1024, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(pool5)
        norm6 = tf.keras.layers.BatchNormalization()(conv6)
        acti6 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(norm6)
        conv6 = tf.keras.layers.Conv2D(1024, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(acti6)
        norm6 = tf.keras.layers.BatchNormalization()(conv6)
        acti6 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(norm6)
        pool6 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(acti6)

        conv7 = tf.keras.layers.Conv2D(2048, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(pool6)
        norm7 = tf.keras.layers.BatchNormalization()(conv7)
        acti7 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(norm7)
        conv7 = tf.keras.layers.Conv2D(2048, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(acti7)
        norm7 = tf.keras.layers.BatchNormalization()(conv7)
        acti7 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(norm7)

        right_up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(acti7), acti6], axis=3)
        right_conv6 = tf.keras.layers.Conv2D(1024, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_up6)
        right_norm6 = tf.keras.layers.BatchNormalization()(right_conv6)
        right_acti6 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_norm6)
        right_conv6 = tf.keras.layers.Conv2D(1024, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_acti6)
        right_norm6 = tf.keras.layers.BatchNormalization()(right_conv6)
        right_acti6 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_norm6)

        right_up5 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(right_acti6), acti5], axis=3)
        right_conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_up5)
        right_norm5 = tf.keras.layers.BatchNormalization()(right_conv5)
        right_acti5 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_norm5)
        right_conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_acti5)
        right_norm5 = tf.keras.layers.BatchNormalization()(right_conv5)
        right_acti5 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_norm5)

        right_up4 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(right_acti5), acti4], axis=3)
        right_conv4 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_up4)
        right_norm4 = tf.keras.layers.BatchNormalization()(right_conv4)
        right_acti4 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_norm4)
        right_conv4 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_acti4)
        right_norm4 = tf.keras.layers.BatchNormalization()(right_conv4)
        right_acti4 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_norm4)

        right_up3 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(right_acti4), acti3], axis=3)
        right_conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_up3)
        right_norm3 = tf.keras.layers.BatchNormalization()(right_conv3)
        right_acti3 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_norm3)
        right_conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_acti3)
        right_norm3 = tf.keras.layers.BatchNormalization()(right_conv3)
        right_acti3 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_norm3)

        right_up2 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(right_acti3), acti2], axis=3)
        right_conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_up2)
        right_norm2 = tf.keras.layers.BatchNormalization()(right_conv2)
        right_acti2 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_norm2)
        right_conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_acti2)
        right_norm2 = tf.keras.layers.BatchNormalization()(right_conv2)
        right_acti2 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_norm2)

        right_up1 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(right_acti2), acti1], axis=3)
        right_conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_up1)
        right_norm1 = tf.keras.layers.BatchNormalization()(right_conv1)
        right_acti1 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_norm1)
        right_conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer=self.kernel_initializer)(right_acti1)
        right_norm1 = tf.keras.layers.BatchNormalization()(right_conv1)
        right_acti1 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_norm1)

        output1 = tf.keras.layers.Conv2D(1, (1, 1))(right_acti1)
        output_norm = tf.keras.layers.BatchNormalization()(output1)
        output = tf.keras.layers.Activation("sigmoid")(output_norm)

        self.model = tf.keras.models.Model(inputs=inputs, outputs=output)


if __name__ == '__main__':
    params = Params('../experiments/batch_norm_toy/params.json')
    net = BiggerLeakyBNUnet(params=params)
    model = net.get_model()
    model.summary()
