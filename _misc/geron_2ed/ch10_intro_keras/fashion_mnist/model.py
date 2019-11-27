import tensorflow as tf
import numpy as np


class DenseNet:

    def __init__(self):
        self.model = None
        self.build_model()
        self.compile_model()

    def build_model(self):
        tf.keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=[28, 28]),
            tf.keras.layers.Dense(300, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")
        ])

    def get_model(self):
        return self.model

    def compile_model(self):
        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer="sgd",
                           metrics=["accuracy"])

    def train(self, X_train, y_train, X_valid, y_valid, epochs=30):
        history = self.model.fit(X_train, y_train,
                                 epochs=epochs,
                                 validation_data=(X_valid, y_valid))
        return history

    def evaluate(self, X_test, y_test):
        test_metrics = self.model.evaluate(X_test, y_test)
        return test_metrics

    def predict(self, X_new):
        y_pred = self.model.predict_classes(X_new)
        return y_pred


if __name__ == '__main__':
    net = DenseNet()
