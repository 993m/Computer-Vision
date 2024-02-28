import os

class Parameters:
    def __init__(self):
        self.base_dir = 'data'
        self.dir_examples_directories = os.path.join(self.base_dir, 'exemplePozitiveDinAdnotari64')
        self.dir_neg_examples = os.path.join(self.base_dir, 'exempleNegative')
        self.dir_test_examples = os.path.join(self.base_dir, 'test/imagini')
        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        self.dim_window = 64
        self.dim_hog_cell = 6
        self.overlap = 0.3
        self.number_negative_examples = 10000
        self.threshold = 0
        self.has_annotations = True
        self.use_flip_images = True
        self.block_size = 4
        self.name = ''




