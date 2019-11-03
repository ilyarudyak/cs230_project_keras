from argparse import Namespace
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = Path.home() / 'data/cats_and_dogs_filtered'
args = Namespace(
    # TODO change to 100
    BATCH_SIZE=3,
    IMG_SHAPE=150,
    train_dir=data_dir / 'train',
    validation_dir=data_dir / 'validation'
)


def get_generators():
    tg = ImageDataGenerator(rescale=1. / 255)
    vg = ImageDataGenerator(rescale=1. / 255)

    train_data_gen = tg.flow_from_directory(batch_size=args.BATCH_SIZE,
                                            directory=args.train_dir,
                                            # TODO change to True
                                            shuffle=False,
                                            target_size=(args.IMG_SHAPE, args.IMG_SHAPE),
                                            class_mode='binary')

    val_data_gen = vg.flow_from_directory(batch_size=args.BATCH_SIZE,
                                          directory=args.validation_dir,
                                          shuffle=False,
                                          target_size=(args.IMG_SHAPE, args.IMG_SHAPE),
                                          class_mode='binary')

    return train_data_gen, val_data_gen
