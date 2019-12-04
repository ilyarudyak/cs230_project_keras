import tensorflow as tf
from utils import Params


class FullUnet:
    """
    This model has the shape specified in the original paper,
    uses UpSampling2D (not Conv2DTranspose) and relu activation
    (not leaky relu). This model uses MaxPool, not BatchNorm.
    """

    def __init__(self,
                 params,
                 set_seed=False):
        self.params = params
        self.input_shape = params.input_shape
        self.set_seed = set_seed
        self.kernel_initializer = params.kernel_initializer

        self.model = None

    def get_model(self):
        if self.model is None:
            self._build_model()
        return self.model

    def _build_model(self):

        if self.set_seed:
            tf.random.set_seed(self.params.seed)

        inputs = tf.keras.layers.Input(self.input_shape)
        conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(inputs)
        conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(conv1)
        pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(pool1)
        conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(conv2)
        pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(pool2)
        conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(conv3)
        pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(pool3)
        conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(conv4)
        pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv4)

        conv5 = tf.keras.layers.Conv2D(1024, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(pool4)
        conv5 = tf.keras.layers.Conv2D(1024, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(conv5)

        up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
        conv6 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(up6)
        conv6 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(conv6)

        up7 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
        conv7 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(up7)
        conv7 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(conv7)

        up8 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
        conv8 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(up8)
        conv8 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(conv8)

        up9 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
        conv9 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(up9)
        conv9 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer=self.kernel_initializer)(conv9)

        conv10 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        self.model = tf.keras.models.Model(inputs=inputs, outputs=conv10)


if __name__ == '__main__':
    params = Params('../experiments/full_unet/params.json')
    net = FullUnet(params=params)
    model = net.get_model()
    model.summary()
