'''DataSequence for model predicting.
Inherit from keras.utils.Sequence,
see https://keras.io/utils/#sequence for more details.
'''

import random
import numpy as np

from keras.utils import Sequence
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array


class DataSequencePred(Sequence):
    def __init__(self, x_raw, batch_size=1, target_size=None):
        '''Create predict generator.

        Args:
            x_raw (np.ndarray | list): an array of image paths, not image data.
            batch_size (int, optional): the sample size of each batch. Defaults to 1.
            target_size (tuple, optional): the size (width, height) of image data. Defaults to None.
        '''
        self.batch_size = batch_size
        self.target_size = target_size
        self.x = x_raw

    def __load_image_data(self, image_path):
        '''Load image data from local image path.

        Args: 
            image_path (str): the image path of an image.

        Returns:
            x (np.ndarray): the image data, known as `x`.
        '''
        img = load_img(image_path, target_size=self.target_size)
        x = img_to_array(img)
        x = preprocess_input(x) # normalization
        return x

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_raw = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.__load_image_data(x) for x in batch_x_raw]
        return np.array(batch_x)
