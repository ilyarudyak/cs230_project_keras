import os
import numpy as np

from skimage.io import imsave, imread
from pathlib import Path
import PIL.Image as Image

data_path = Path.home() / 'data/ultrasound_nerve/'

image_rows = 420
image_cols = 580


def create_train_data(data_path=Path.home() / 'data/ultrasound_nerve/'):
    train_data_path = os.path.join(str(data_path), 'train')
    images = os.listdir(train_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:

        if 'mask' in image_name:
            continue

        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = Image.open(os.path.join(train_data_path, image_name))
        img_mask = Image.open(os.path.join(train_data_path, image_mask_name))

        imgs[i, :, :] = np.array(img)
        imgs_mask[i, :, :] = np.array(img_mask)

        if i % 1000 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(str(data_path),'imgs_train.npy'), imgs)
    np.save(os.path.join(str(data_path),'imgs_mask_train.npy'), imgs_mask)
    print('Saving to .npy files done.')


def load_train_data(data_path=Path.home() / 'data/ultrasound_nerve/'):
    imgs_train = np.load(data_path / 'imgs_train.npy')
    imgs_mask_train = np.load(data_path / 'imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data(data_path=Path.home() / 'data/ultrasound_nerve/'):
    train_data_path = os.path.join(str(data_path), 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = Image.open(os.path.join(train_data_path, image_name))

        imgs[i, :, :] = np.array(img)
        imgs_id[i] = np.array(img_id)

        if i % 1000 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(str(data_path / 'imgs_test.npy'), imgs)
    np.save(str(data_path / 'imgs_id_test.npy'), imgs_id)
    print('Saving to .npy files done.')


def load_test_data(data_path=Path.home() / 'data/ultrasound_nerve/'):
    imgs_test = np.load(data_path / 'imgs_test.npy')
    imgs_id = np.load(data_path / 'imgs_id_test.npy')
    return imgs_test, imgs_id


if __name__ == '__main__':
    create_train_data()
    create_test_data()
