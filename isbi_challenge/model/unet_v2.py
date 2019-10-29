from keras.layers import MaxPool2D, Conv2D, Input, Dropout, Conv2DTranspose, concatenate
from keras import Model


class Unet:

    def __init__(self, params):

        self.input_shape = params.input_shape
        self.kernel_initializer = params.kernel_initializer
        self.init_filters = params.init_filters
        self.dropout = params.dropout

        # no tuning for these parameters
        self.config = {
            'kernel_size': (3, 3),
            'activation': 'relu',
            'padding': 'same',
            'pool_size': (2, 2),
            'kernel_size_transposed': (2, 2),
            'strides_transposed': (2, 2)
        }

    def contracting_block(self, filters, in_layer, is_max_pool=True):

        # no max pool for input layer
        if is_max_pool:
            max_pool = MaxPool2D(pool_size=self.config['pool_size'])(in_layer)
        else:
            max_pool = in_layer

        conv_layer = self.get_conv2D(filters=filters)(max_pool)
        conv_layer = self.get_conv2D(filters=filters)(conv_layer)

        conv_layer = Dropout(rate=self.dropout)(conv_layer)
        return conv_layer

    def expanding_block(self, filters, in_layer, concat_layer, is_max_pool=False):

        if is_max_pool:
            max_pool = MaxPool2D(pool_size=self.config['pool_size'])(in_layer)
        else:
            max_pool = in_layer

        conv_layer = self.get_conv2D(filters=filters)(max_pool)
        conv_layer = self.get_conv2D(filters=filters)(conv_layer)

        conv_trans_layer = self.get_con2D_trans(filters=filters//2)(conv_layer)

        expand_layer = concatenate([conv_trans_layer, concat_layer])
        expand_layer = Dropout(rate=self.dropout)(expand_layer)

        return expand_layer

    def final_block(self, in_layer):

        conv_layer = self.get_conv2D(filters=self.init_filters)(in_layer)
        conv_layer = self.get_conv2D(filters=self.init_filters)(conv_layer)

        output_layer = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(conv_layer)

        return output_layer

    def get_conv2D(self, filters):
        return Conv2D(filters=filters,
                      kernel_size=self.config['kernel_size'],
                      activation=self.config['activation'],
                      padding=self.config['padding'],
                      kernel_initializer=self.kernel_initializer)

    def get_con2D_trans(self, filters):
        return Conv2DTranspose(filters=filters,
                               kernel_size=self.config['kernel_size_transposed'],
                               strides=self.config['strides_transposed'],
                               activation=self.config['activation'],
                               kernel_initializer=self.kernel_initializer)

    def build_model(self):

        input_layer = Input(shape=self.input_shape, dtype='float32')

        conv1 = self.contracting_block(filters=self.init_filters, in_layer=input_layer, is_max_pool=False)
        conv2 = self.contracting_block(filters=self.init_filters * 2, in_layer=conv1)
        conv3 = self.contracting_block(filters=self.init_filters * 4, in_layer=conv2)
        conv4 = self.contracting_block(filters=self.init_filters * 8, in_layer=conv3)

        expand1 = self.expanding_block(filters=self.init_filters * 16, in_layer=conv4, concat_layer=conv4, is_max_pool=True)
        expand2 = self.expanding_block(filters=self.init_filters * 8, in_layer=expand1, concat_layer=conv3)
        expand3 = self.expanding_block(filters=self.init_filters * 4, in_layer=expand2, concat_layer=conv2)
        expand4 = self.expanding_block(filters=self.init_filters * 2, in_layer=expand3, concat_layer=conv1)

        output_layer = self.final_block(expand4)

        model = Model(input_layer, output_layer)

        return model
