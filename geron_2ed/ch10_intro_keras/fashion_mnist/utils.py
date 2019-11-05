import matplotlib.pyplot as plt
import pandas as pd


def plot_sample(X_train, y_train, n_rows=4, n_cols=10):
    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
            plt.axis('off')
            plt.title(get_class_names(y_train[index]), fontsize=12)
    # the amount of width reserved for space between subplots
    # the amount of height reserved for space between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.5)


def get_class_names(index):
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    return class_names[index]


def plot_metrics(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
