import tensorflow as tf
from pathlib import Path
import utils


class SkinLesionDataGen:

    def __init__(self,
                 params,
                 image_color_mode='rgb',
                 mask_color_mode='grayscale',
                 image_classes=('images',),
                 mask_classes=('masks',),
                 data_dir=Path.home() / 'data/isic_2018',
                 ):
        self.params = params
        self.batch_size = self.params.batch_size
        self.target_size = self.params.input_shape[0:2]
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
        # we do NOT augment val data
        self.val_aug_dict = dict(
            rescale=1. / 255
        )

        # we create separate image and mask generators
        # using flow_from_directory so we have to specify classes
        self.image_classes = list(image_classes)
        self.mask_classes = list(mask_classes)

        self.train_dir = data_dir / 'train'
        self.val_dir = data_dir / 'val'

        self.train_gen = None
        self.val_gen = None

    def get_train_gen(self):

        if self.train_gen is None:
            self.train_gen = self._build_gen(split='train')

        return self.train_gen

    def get_val_gen(self):

        if self.val_gen is None:
            self.val_gen = self._build_gen(split='val')

        return self.val_gen

    def _build_gen(self, split):

        if split == 'train':
            # we set the same set to get identical augmentation
            # of an image and its mask
            aug_dict = self.train_aug_dict
            data_dir = self.train_dir
        elif split == 'val':
            aug_dict = self.val_aug_dict
            data_dir = self.val_dir
        else:
            raise ValueError('wrong split')

        image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)
        mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)

        image_generator = image_datagen.flow_from_directory(
            data_dir,
            classes=self.image_classes,
            class_mode=None,
            color_mode=self.image_color_mode,
            target_size=self.target_size,
            batch_size=self.batch_size,
            seed=self.seed)
        mask_generator = mask_datagen.flow_from_directory(
            data_dir,
            classes=self.mask_classes,
            class_mode=None,
            color_mode=self.mask_color_mode,
            target_size=self.target_size,
            batch_size=self.batch_size,
            seed=self.seed)

        generator = zip(image_generator, mask_generator)
        for (img, mask) in generator:
            # mask will contain only 0s and 1s
            mask = mask.round()
            yield img, mask


if __name__ == '__main__':
    data_dir = Path('../experiments/bigger_leaky_unet_toy')
    params = utils.Params(data_dir / 'params.json')
    gen = SkinLesionDataGen(params=params)
    print(gen.target_size)

