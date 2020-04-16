'''DataSequence for model training.
Inherit from keras.utils.Sequence,
see https://keras.io/utils/#sequence for more details.
'''

import random
import numpy as np

from keras.utils import Sequence
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array


class DataSequence(Sequence):
    def __init__(self, x_raw, y_raw, batch_size, target_size=(224, 224)):
        '''Create train / validate generator.

        Args:
            x_raw (np.ndarray): an array of image paths, not image data.
            y_raw (np.ndarray): an array of image classes.
            batch_size (int): the sample size of each batch.
            target_size (tuple, optional): the size (width, height) of image data. Defaults to (224, 224).
        '''
        self.batch_size = batch_size
        self.target_size = target_size
        self.x, self.y = self.__create_pairs(x_raw, y_raw)

    def __create_pairs(self, x_raw, y_raw):
        '''Create positive and negative pairs.
        Wont load image data itself but create a list of sample pairs,
        so that we can train model with generator and load image data asynchronous.

        Args:
            x_raw (np.ndarray): an array of image paths, not image data.
            y_raw (np.ndarray): an array of image classes.

        Returns:
            pairs (list): the samples of image paths combined into pairs, known as `x`.
            labels (list): the samples of image labels combined into pairs, known as `y`.
        '''
        pairs = []
        labels = []
        classes = np.unique(y_raw) # sorted already
        digit_indices = [np.where(y_raw == i)[0] for i in classes]
        for d in classes:
            classes_neg = classes[classes != d]
            for i in range(len(digit_indices[d])-1):
                # positive pair
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                pairs += [[x_raw[z1], x_raw[z2]]]
                # negative pair
                dn = random.choice(classes_neg)
                j = random.randrange(len(digit_indices[dn]))
                z1, z2 = digit_indices[d][i], digit_indices[dn][j]
                pairs += [[x_raw[z1], x_raw[z2]]]
                labels += [1, 0]
        return pairs, labels

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
        batch_y_raw = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x1, batch_x2 = [], []
        batch_y = []
        for (x1, x2), y in zip(batch_x_raw, batch_y_raw):
            try:
                image_x1 = self.__load_image_data(x1)
                image_x2 = self.__load_image_data(x2)
                batch_x1.append(image_x1)
                batch_x2.append(image_x2)
                batch_y.append(y)
            except UserWarning as w:
                pass
            except Exception as e:
                pass

        batch_x1 = np.array(batch_x1)
        batch_x2 = np.array(batch_x2)
        batch_x = [batch_x1, batch_x2]
        batch_y = np.array(batch_y)
        return batch_x, batch_y
