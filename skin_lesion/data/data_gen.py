import tensorflow as tf
from pathlib import Path


class SkinLesionDataGen:

    def __init__(self,
                 params,
                 image_color_mode='rgb',
                 mask_color_mode='grayscale',
                 image_classes=('images',),
                 mask_classes=('masks',),
                 train_dir=Path.home() / 'data/isic_2018/train',
                 val_dir=Path.home() / 'data/isic_2018/val'
                 ):
        self.params = params
        self.batch_size = self.params.batch_size
        self.target_size = self.params.target_size
        self.seed = self.params.seed

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        self.train_aug_dict = dict(
            rescale=1./255,
            rotation_range=180,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

        # we create separate image and mask generators
        # using flow_from_directory so we have to specify classes
        self.image_classes = list(image_classes)
        self.mask_classes = list(mask_classes)

        self.train_dir = train_dir
        self.val_dir = val_dir

        self.train_gen = None
        self.val_gen = None

    def get_train_gen(self):

        if self.train_gen is None:
            self.train_gen = self._build_train_gen()

        return self.train_gen

    def _build_train_gen(self):

        # we set the same set to get identical augmentation
        # of an image and its mask
        image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**self.train_aug_dict)
        mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**self.train_aug_dict)
        image_generator = image_datagen.flow_from_directory(
            self.train_dir,
            classes=self.image_classes,
            class_mode=None,
            color_mode=self.image_color_mode,
            target_size=self.target_size,
            batch_size=self.batch_size,
            seed=self.seed)
        mask_generator = mask_datagen.flow_from_directory(
            self.train_dir,
            classes=self.mask_classes,
            class_mode=None,
            color_mode=self.mask_color_mode,
            target_size=self.target_size,
            batch_size=self.batch_size,
            seed=self.seed)
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            # mask will contain only 0s and 1s
            mask = mask.round()
            yield img, mask

