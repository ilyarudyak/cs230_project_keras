import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


class ConvNet:

    def __init__(self):
        self.model = None
        self.build_model()

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

    def get_model(self):
        return self.model


class MobileNetTransf:

    def __init__(self):

        self.config = {
            'CLASSIFIER_URL': 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2',
            'IMAGE_RES': 224
        }

        self.model = None
        self.build_model()

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            hub.KerasLayer(self.config['CLASSIFIER_URL'],
                           input_shape=(self.config['IMAGE_RES'], self.config['IMAGE_RES'], 3))
        ])

    def get_model(self):
        return self.model
