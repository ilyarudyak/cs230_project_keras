import tensorflow as tf
from utils import Params


class BiggerLeakyUnet:
    """
    ...
    """

    def __init__(self,
                 params,
                 set_seed=False):
        self.params = params
        self.input_shape = params.input_shape
        self.alpha = params.alpha
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
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(inputs)
        acti1 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(conv1)
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(acti1)
        acti1 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(acti1)

        conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(pool1)
        acti2 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(conv2)
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(acti2)
        acti2 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(acti2)

        conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(pool2)
        acti3 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(conv3)
        conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(acti3)
        acti3 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(acti3)

        conv4 = tf.keras.layers.Conv2D(256, (3, 3), padding="same")(pool3)
        acti4 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(conv4)
        conv4 = tf.keras.layers.Conv2D(256, (3, 3), padding="same")(acti4)
        acti4 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(conv4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(acti4)

        conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(pool4)
        acti5 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(conv5)
        conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(acti5)
        acti5 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(conv5)
        pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(acti5)

        conv6 = tf.keras.layers.Conv2D(1024, (3, 3), padding="same")(pool5)
        acti6 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(conv6)
        conv6 = tf.keras.layers.Conv2D(1024, (3, 3), padding="same")(acti6)
        acti6 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(conv6)
        pool6 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(acti6)

        conv7 = tf.keras.layers.Conv2D(2048, (3, 3), padding="same")(pool6)
        acti7 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(conv7)
        conv7 = tf.keras.layers.Conv2D(2048, (3, 3), padding="same")(acti7)
        acti7 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(conv7)

        right_up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(acti7), acti6], axis=3)
        right_conv6 = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(right_up6)
        right_acti6 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv6)
        right_conv6 = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(right_acti6)
        right_acti6 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv6)

        right_up5 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(right_acti6), acti5], axis=3)
        right_conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(right_up5)
        right_acti5 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv5)
        right_conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding="same")(right_acti5)
        right_acti5 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv5)

        right_up4 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(right_acti5), acti4], axis=3)
        right_conv4 = tf.keras.layers.Conv2D(256, (3, 3), padding="same")(right_up4)
        right_acti4 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv4)
        right_conv4 = tf.keras.layers.Conv2D(256, (3, 3), padding="same")(right_acti4)
        right_acti4 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv4)

        right_up3 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(right_acti4), acti3], axis=3)
        right_conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(right_up3)
        right_acti3 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv3)
        right_conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(right_acti3)
        right_acti3 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv3)

        right_up2 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(right_acti3), acti2], axis=3)
        right_conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(right_up2)
        right_acti2 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv2)
        right_conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(right_acti2)
        right_acti2 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv2)

        right_up1 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(right_acti2), acti1], axis=3)
        right_conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(right_up1)
        right_acti1 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv1)
        right_conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(right_acti1)
        right_acti1 = tf.keras.layers.LeakyReLU(alpha=self.alpha)(right_conv1)

        output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(right_acti1)

        self.model = tf.keras.models.Model(inputs=inputs, outputs=output)


if __name__ == '__main__':
    params = Params('../experiments/bigger_leaky_unet/params.json')
    net = BiggerLeakyUnet(params=params)
    model = net.get_model()
    model.summary()
