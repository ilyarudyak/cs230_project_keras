from argparse import Namespace

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import pathlib

data_dir = pathlib.Path.home() / 'data/isic_2018'
args = Namespace(
    image_dir=data_dir / 'images',
    mask_dir=data_dir / 'masks',
    dirs={
        'train_images': data_dir / 'train/images',
        'train_masks': data_dir / 'train/masks',
        'val_images': data_dir / 'val/images',
        'val_masks': data_dir / 'val/masks',
        'test_images': data_dir / 'test/images',
        'test_masks': data_dir / 'test/masks',
    },
    splits=['train', 'val', 'test']
)


#######################################################
##################### show images #####################
#######################################################


def show_images_from_path(image_dir=args.image_dir, m=4, n=4):
    paths = sorted(list(image_dir.glob('*.jpg')))[:m * n]
    plot_images(paths, m, n)


def show_masks_from_path(mask_dir=args.mask_dir, m=4, n=4):
    paths = sorted(list(mask_dir.glob('*.png')))[:m * n]
    plot_images(paths, m, n)


def plot_images(paths, m, n):
    _, ax = plt.subplots(m, n)
    for i in range(m):
        for j in range(n):
            path = paths[i * n + j]
            img = tf.keras.preprocessing.image.load_img(path)
            ax[i, j].imshow(img)
            ax[i, j].axis('off')


#######################################################
##################### copy files ######################
#######################################################

def copy_files(image_dir=args.image_dir):
    """
    Copy files from common images and masks folders
    into directories specified in args.dirs.
    """
    all_paths = get_split_filenames(image_dir)
    for paths, split in zip(all_paths, args.splits):
        copy_files_split(paths=paths, split=split)


def get_split_filenames(image_dir=args.image_dir, ratio=.4, val_test_ratio=.5, seed=42):
    paths = np.array(sorted(list(image_dir.glob('*.jpg'))))
    train_paths, val_test_paths = train_test_split(paths,
                                                   test_size=ratio,
                                                   random_state=seed)
    val_paths, test_paths = train_test_split(val_test_paths,
                                             test_size=val_test_ratio,
                                             random_state=seed)
    return train_paths, val_paths, test_paths


def get_mask_filepath(image_path, mask_dir=args.mask_dir):
    mask_filename_short = str(image_path).split('/')[-1].split('.')[0] + '_segmentation.png'
    mask_path = mask_dir / pathlib.Path(mask_filename_short)
    return mask_path


def copy_files_split(paths, dirs=args.dirs, split='train'):
    for image_path in paths:
        shutil.copy(image_path, dirs[split + '_images'])
        mask_path = get_mask_filepath(image_path)
        shutil.copy(mask_path, dirs[split + '_masks'])


def get_filename_from_path(path):
    return str(path).split('/')[-1]


def get_image_filename_from_mask_path(path):
    return str(path).split('/')[-1][:12] + '.jpg'


if __name__ == '__main__':
    copy_files()



