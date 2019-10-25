from PIL import Image
import numpy as np
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path


class ISBI2012:

    def __init__(self,
                 data_dir=Path.home() / 'data/isbi2012',
                 validation_set_pages=(5, 15, 25),
                 test_set_pages=(0, 10, 20)):

        self.seed = 42

        self.image_path = data_dir / 'train-volume.tif'
        self.mask_path = data_dir / 'train-labels.tif'

        self.validation_set_pages = set(validation_set_pages)
        self.test_set_pages = set(test_set_pages)
        self.page_count = 30
        self.training_set_pages = set(range(self.page_count)) - self.validation_set_pages - self.test_set_pages

        self.augmentation = {
            'rotation_range': 0.2,
            'width_shift_range': 0.05,
            'height_shift_range': 0.05,
            'shear_range': 0.05,
            'zoom_range': 0.05,
            'horizontal_flip': True,
            'fill_mode': 'nearest'
        }

        self.image_data = None
        self.mask_data = None
        self.load_data()

    def load_data(self):
        # if self.image_data is None or self.label_data is None:
        image_tif_volume = Image.open(self.image_path)
        mask_tif_volume = Image.open(self.mask_path)
        image_shape = (self.page_count, image_tif_volume.size[0], image_tif_volume.size[1], 1)
        mask_shape = (self.page_count, mask_tif_volume.size[0], mask_tif_volume.size[1], 1)

        self.image_data = np.empty(image_shape, dtype=float)
        self.mask_data = np.empty(mask_shape, dtype=float)

        for i in range(self.page_count):
            image_tif_volume.seek(i)
            mask_tif_volume.seek(i)
            self.image_data[i] = np.array(image_tif_volume.getdata()).reshape(image_shape[1:])
            self.mask_data[i] = np.array(mask_tif_volume.getdata()).reshape(mask_shape[1:])

    def generator(self, mode='training', batch_size=1):
        """
        Returns a generator for batches of images and corresponded labels.
        Generator yields a pair (image_batch[batch_size, image_height, image_width, channels],
        label_batch[batch_size, label_height, label_width, 1]),
        normalized to [0.0, 1.0] range.
        Training set uses image augmentation. Validation and test sets do
        *not* use any augmentation.
        """

        augmentation = {'rescale': 1. / 255}
        if mode == 'training':
            augmentation.update(self.augmentation)
            pages = self.training_set_pages
        elif mode == 'validation':
            pages = self.validation_set_pages
        elif mode == 'test':
            pages = self.test_set_pages
        else:
            raise ValueError('Unknown dataset mode ', mode)

        number_of_images = len(pages)
        image_shape = (number_of_images, self.image_data.shape[1], self.image_data.shape[2], 1)
        mask_shape = (number_of_images, self.mask_data.shape[1], self.mask_data.shape[2], 1)
        image_data = np.empty(image_shape, dtype=float)
        mask_data = np.empty(mask_shape, dtype=float)

        for i, p in enumerate(pages):
            image_data[i] = self.image_data[p]
            mask_data[i] = self.mask_data[p]

        image_gen = ImageDataGenerator(**augmentation)
        mask_gen = ImageDataGenerator(**augmentation)

        image_gen.fit(image_data, augment=True)
        mask_gen.fit(mask_data, augment=True)
        # set seed to get identical augmentation of an image and its mask
        image_generator = image_gen.flow(x=image_data, batch_size=batch_size, seed=self.seed)
        mask_generator = mask_gen.flow(x=mask_data, batch_size=batch_size, seed=self.seed)

        data_generator = zip(image_generator, mask_generator)
        for image, mask in data_generator:
            yield image, mask
