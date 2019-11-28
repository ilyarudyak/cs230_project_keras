import unittest
from data import data_utils


class TestDataUtils(unittest.TestCase):

    def setUp(self):
        all_paths = data_utils.get_split_filenames()
        self.all_filenames_sorted = [list(map(data_utils.get_filename_from_path, sorted(paths)))
                                     for paths in all_paths]
        self.correct_count = [1556, 519, 519]

    def test_all_images(self):
        """
        test that we actually copied files specified in our splits.
        """
        for i, split in enumerate(data_utils.args.splits):
            paths_actual = data_utils.args.dirs[split + '_images'].glob('*')
            filenames_actual_sorted = list(map(data_utils.get_filename_from_path, sorted(paths_actual)))
            self.assertEqual(filenames_actual_sorted, self.all_filenames_sorted[i])

    def test_count(self):
        """
        test that we copied correct number of files.
        """
        for i, split in enumerate(data_utils.args.splits):
            image_count = len(list(data_utils.args.dirs[split + '_images'].glob('*')))
            mask_count = len(list(data_utils.args.dirs[split + '_masks'].glob('*')))
            self.assertEqual(image_count, mask_count)
            self.assertEqual(image_count, self.correct_count[i])

    def test_all_masks(self):
        """
        test that we have masks exactly for image files in each split.
        """
        for i, split in enumerate(data_utils.args.splits):
            paths_images_actual = data_utils.args.dirs[split + '_images'].glob('*')
            paths_masks_actual = data_utils.args.dirs[split + '_masks'].glob('*')

            filenames_images_actual_sorted = list(map(data_utils.get_filename_from_path, sorted(paths_images_actual)))
            filenames_masks_actual_sorted = list(map(data_utils.get_image_filename_from_mask_path,
                                                     sorted(paths_masks_actual)))

            self.assertEqual(filenames_images_actual_sorted, filenames_masks_actual_sorted)


if __name__ == '__main__':
    unittest.main()
