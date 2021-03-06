from argparse import Namespace
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = Path.home() / 'data/cats_and_dogs_filtered'
args = Namespace(
    IMG_SHAPE=150,
    train_dir=data_dir / 'train',
    validation_dir=data_dir / 'validation'
)


def get_generators(batch_size):
    tg = ImageDataGenerator(rescale=1. / 255)
    vg = ImageDataGenerator(rescale=1. / 255)

    train_data_gen = tg.flow_from_directory(batch_size=batch_size,
                                            directory=args.train_dir,
                                            shuffle=True,
                                            target_size=(args.IMG_SHAPE, args.IMG_SHAPE),
                                            class_mode='binary')

    val_data_gen = vg.flow_from_directory(batch_size=batch_size,
                                          directory=args.validation_dir,
                                          shuffle=False,
                                          target_size=(args.IMG_SHAPE, args.IMG_SHAPE),
                                          class_mode='binary')

    return train_data_gen, val_data_gen


def get_generators_debug(batch_size):
    tg = ImageDataGenerator(
                            rescale=1./255,
                            rotation_range=40,
                            # width_shift_range=0.2,
                            # height_shift_range=0.2,
                            # shear_range=0.2,
                            # zoom_range=0.2,
                            # horizontal_flip=True,
                            fill_mode='nearest'
    )
    vg = ImageDataGenerator(rescale=1. / 255)

    train_data_gen = tg.flow_from_directory(batch_size=batch_size,
                                            directory=args.train_dir,
                                            shuffle=True,
                                            target_size=(args.IMG_SHAPE, args.IMG_SHAPE),
                                            class_mode='binary')

    val_data_gen = vg.flow_from_directory(batch_size=batch_size,
                                          directory=args.validation_dir,
                                          shuffle=False,
                                          target_size=(args.IMG_SHAPE, args.IMG_SHAPE),
                                          class_mode='binary')

    return train_data_gen, val_data_gen


def get_generators_aug(batch_size):
    tg = ImageDataGenerator(
                            rescale=1./255,
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest'
    )
    vg = ImageDataGenerator(rescale=1. / 255)

    train_data_gen = tg.flow_from_directory(batch_size=batch_size,
                                            directory=args.train_dir,
                                            shuffle=True,
                                            target_size=(args.IMG_SHAPE, args.IMG_SHAPE),
                                            class_mode='binary')

    val_data_gen = vg.flow_from_directory(batch_size=batch_size,
                                          directory=args.validation_dir,
                                          shuffle=False,
                                          target_size=(args.IMG_SHAPE, args.IMG_SHAPE),
                                          class_mode='binary')

    return train_data_gen, val_data_gen
