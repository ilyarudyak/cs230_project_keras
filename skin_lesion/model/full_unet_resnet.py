import tensorflow as tf
from utils import Params


class FullUnetResnet:
    """
    build upon:
    https://github.com/killthekitten/kaggle-carvana-2017/blob/master/models.py
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

        self.base_model = tf.keras.applications.ResNet50(weights='imagenet',
                                                         include_top=False,
                                                         input_shape=params.input_shape)
        self.vgg_model = tf.keras.applications.VGG16(weights='imagenet',
                                                     include_top=False,
                                                     input_tensor=self.base_model.input,
                                                     input_shape=params.input_shape)

        # naming convention is changed in tf.keras
        self.resnet_config = {
            'activation_1': 'conv1_relu',
            'activation_10': 'conv2_block3_out',
            'activation_22': 'conv3_block4_out',
            'activation_40': 'conv4_block6_out',
            'activation_49': 'conv5_block3_out'
        }

        if freeze:
            self.freeze_layers()

        self.model = None

    def freeze_layers(self):
        for layer in self.base_model.layers:
            layer.trainable = False

        for layer in self.vgg_model.layers:
            layer.trainable = False

    def get_model(self):
        if self.model is None:
            self._build_model()

        return self.model

    def _build_model(self):

        if self.set_seed:
            tf.random.set_seed(self.params.seed)

        conv1 = self.base_model.get_layer(self.resnet_config['activation_1']).output
        conv2 = self.base_model.get_layer(self.resnet_config['activation_10']).output
        conv3 = self.base_model.get_layer(self.resnet_config['activation_22']).output
        conv4 = self.base_model.get_layer(self.resnet_config['activation_40']).output
        conv5 = self.base_model.get_layer(self.resnet_config['activation_49']).output

        up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv5), conv4], axis=-1)
        conv6 = self.conv_block_simple(up6, 256, "conv6_1")
        conv6 = self.conv_block_simple(conv6, 256, "conv6_2")

        up7 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv6), conv3], axis=-1)
        conv7 = self.conv_block_simple(up7, 192, "conv7_1")
        conv7 = self.conv_block_simple(conv7, 192, "conv7_2")

        up8 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv7), conv2], axis=-1)
        conv8 = self.conv_block_simple(up8, 128, "conv8_1")
        conv8 = self.conv_block_simple(conv8, 128, "conv8_2")

        up9 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv8), conv1], axis=-1)
        conv9 = self.conv_block_simple(up9, 64, "conv9_1")
        conv9 = self.conv_block_simple(conv9, 64, "conv9_2")

        vgg_first_conv = self.vgg_model.get_layer("block1_conv2").output
        up10 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv9), self.base_model.input, vgg_first_conv], axis=-1)
        conv10 = self.conv_block_simple(up10, 32, "conv10_1")
        conv10 = self.conv_block_simple(conv10, 32, "conv10_2")
        conv10 = tf.keras.layers.SpatialDropout2D(0.2)(conv10)

        output = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)

        self.model = tf.keras.models.Model(inputs=self.base_model.input, outputs=output)

    def conv_block_simple(self, prev_layer, filters, prefix, strides=(1, 1)):
        conv = tf.keras.layers.Conv2D(filters, (3, 3),
                                      padding="same",
                                      kernel_initializer=self.kernel_initializer,
                                      strides=strides,
                                      name=prefix + "_conv")(prev_layer)
        conv = tf.keras.layers.LeakyReLU(alpha=self.alpha, name=prefix + "_activation")(conv)
        return conv


if __name__ == '__main__':
    params = Params('../experiments/transf_learn_resnet_toy/params.json')
    net = FullUnetResnet(params=params)
    model = net.get_model()
    model.summary()
