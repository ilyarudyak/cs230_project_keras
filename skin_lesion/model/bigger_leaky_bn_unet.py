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


if __name__ == '__main__':
    params = Params('../experiments/bigger_leaky_unet/params.json')
    net = BiggerLeakyBNUnet(params=params)
    model = net.get_model()
    model.summary()
