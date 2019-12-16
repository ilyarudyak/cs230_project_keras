import tensorflow as tf
from utils import Params


class FullUnetResnet:
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

        self.base_model = tf.keras.applications.ResNet50(weights='imagenet',
                                                         include_top=False,
                                                         input_shape=params.input_shape)
        self.vgg_model = tf.keras.applications.VGG16(weights='imagenet',
                                                     include_top=False,
                                                     input_shape=params.input_shape)

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

        x = self.base_model.output

        bottom = self.double_block(x, 1024, prefix='bottom')

        # 14 x 14 block5
        left_block5 = self.base_model.get_layer('conv4_block6_out').output
        right_conv5 = self.up_sampling_block(bottom, left_block5, 512, 'right_conv5')

        # 28 x 28 block4
        left_block4 = self.base_model.get_layer('conv3_block4_out').output
        right_conv4 = self.up_sampling_block(right_conv5, left_block4, 256, 'right_conv4')

        # 56 x 56 block3
        left_block3 = self.base_model.get_layer('conv2_block3_out').output
        right_conv3 = self.up_sampling_block(right_conv4, left_block3, 128, 'right_conv3')

        # 112 x 112 block2
        left_block2 = self.base_model.get_layer('conv1_relu').output
        right_conv2 = self.up_sampling_block(right_conv3, left_block2, 64, 'right_conv2')

        # Resnet doesn't have 224 x 224 layer conv, use first conv layer in VGG16 instead
        vgg_first_conv = self.vgg_model.get_layer("block1_conv2").output
        right_conv1 = self.up_sampling_block(right_conv2, vgg_first_conv, 32, 'right_conv1')
        right_dropout1 = tf.keras.layers.SpatialDropout2D(0.2)(right_conv1)

        # Sigmoid
        output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(right_dropout1)
        self.model = tf.keras.models.Model(inputs=self.base_model.input, outputs=output)

    def conv_block(self, prevlayer, filters, prefix, strides=(1, 1)):
        conv = tf.keras.layers.Conv2D(filters, (3, 3),
                                      padding="same",
                                      kernel_initializer="he_normal",
                                      strides=strides,
                                      name=prefix + "_conv")(prevlayer)
        # if batch_norm:
        #     conv = BatchNormalization(name=prefix + "_bn")(conv)
        # if layer_norm:
        #     conv = LayerNormalization()(conv)
        conv = tf.keras.layers.LeakyReLU(alpha=0.001,
                                         name=prefix + "_activation")(conv)
        return conv

    def double_block(self, prevlayer, filters, prefix, strides=(1, 1)):
        layer1 = self.conv_block(prevlayer, filters, prefix + "1", strides)
        layer2 = self.conv_block(layer1, filters, prefix + "2", strides)
        return layer2

    def up_sampling_block(self, up_sampling_layer, left_skip_layer, filters, prefix, strides=(1, 1)):
        up_layer = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(up_sampling_layer), left_skip_layer], axis=3)
        double_block_layer = self.double_block(up_layer, filters, prefix, strides)
        return double_block_layer


if __name__ == '__main__':
    params = Params('../experiments/transf_learn_resnet_toy/params.json')
    net = FullUnetResnet(params=params)
    model = net.get_model()
    model.summary()
