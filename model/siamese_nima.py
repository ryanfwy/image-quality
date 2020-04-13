'''Siamese NIMA model for training, testing and predicting.'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').disabled = True

import numpy as np
import pandas as pd

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from keras import backend as K

from model.data_sequence import DataSequence
from model.data_sequence_pred import DataSequencePred


class SiameseNIMA():
    def __init__(self,
                 input_shape=(None, None, 3),
                 output_dir=None):
        '''Siamese NIMA.

        Args:
            input_shape (tuple, optional): the input shape of Siamese NIMA.
                Defaults to (None, None, 3).
            output_dir (str, optional): the directory to save every output files.
                Defaults to None, save to current direcotry.
        '''
        self.input_shape = input_shape

        self._output_dir = output_dir or os.path.join(os.path.dirname(__file__), '..')
        self._checkpoint_dir = os.path.join(self._output_dir, 'checkpoints')

        self._model_nima = None
        self._model_siamese = None

    @property
    def model_nima(self):
        '''NIMA network only, generally used to evaluate image quality.'''
        if self._model_nima is None:
            print('Warning: you should first `train()`, `predict()` or `bulid()` the model.')
        return self._model_nima

    # @property
    # def model_siamese(self):
    #     '''Siamese NIMA network, generally used to train NIMA network.'''
    #     return self._model_siamese

    @staticmethod
    def _nima_mean_layer(input_tensor):
        '''Calculate the mean score of NIMA model's output.'''
        si  = K.arange(1., 11.)
        return K.sum(input_tensor * si, axis=1, keepdims=True)

    @staticmethod
    def _distance_layer(input_tensors):
        '''Calculate the l2 distacne of two inputs.'''
        x1, x2 = input_tensors
        sum_square = K.sum(K.square(x1 - x2), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    @staticmethod
    def _contrastive_loss(y_true, y_pred):
        '''Contrastive loss that is used to train Siamese NIMA.
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        Args:
            y_true (np.ndarray): an array of ground truth labels, known as `y`.
            y_pred (np.ndarray): an array of predict labels, known as `y_hat`

        Returns:
            Result of the contrastive loss.
        '''
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def _build_nima_network(self, weight_path=None, layer_to_freeze=618):
        '''Build NIMA (Neural Image Assessment) network.
        https://arxiv.org/abs/1709.05424

        Args:
            weight_path (str, optional): the file path of NIMA network weight.
                If None, no network weight will be loaded, known as retraining.
                Otherwise, NIMA weight will be loaded, known as fine-tuning.
                Defaults to None.
            layer_to_freeze (int, optional): the last number of layer to freeze.
                This will be used when `weight_path` is passed.
                If None, no layer will be freezed.
                Defaults to 618.

        Returns:
            model (keras.Model): NIMA model.
        '''
        # NIMA model
        base_model = InceptionResNetV2(input_shape=self.input_shape,
                                       include_top=False,
                                       pooling='avg',
                                       weights=None)
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)
        model = Model(base_model.input, x)

        # load weights
        if weight_path and os.path.isfile(weight_path):
            print('Loading nima weight:', weight_path)
            model.load_weights(weight_path)
        else:
            print('Loading nima weight: None')
            layer_to_freeze = 0

        # freeze layers
        if layer_to_freeze:
            print('Freezing layers: < #{}'.format(layer_to_freeze))
            for layer in model.layers[:layer_to_freeze]:
                layer.trainable = False
            for layer in model.layers[layer_to_freeze:]:
                layer.trainable = True

        return model

    def _build_siamese_network(self, base_model):
        '''Build Siamese network with `base_model` to be shared.

        Args:
            base_model (keras): base model network to be shared.
                Here, NIMA network would be used.

        Returns:
            model (keras.Model): Siamese NIMA model.
        '''
        # bulid siamese network
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)
        base1 = base_model(input_a)
        base2 = base_model(input_b)
        processed_a = Lambda(self._nima_mean_layer)(base1)
        processed_b = Lambda(self._nima_mean_layer)(base2)
        # difine distance layer
        distance = Lambda(self._distance_layer)([processed_a, processed_b])
        model = Model([input_a, input_b], distance)
        return model

    def _checkpoint_on_epoch_end(self, epoch, logs, file_name_prefix='nima_weights'):
        '''Callback function that save model weights at checkpoints.'''
        if 'val_accuracy' in logs:
            file_name = '{file_name_prefix}{epoch:02d}-{val_accuracy:.3f}.h5'.format(
                file_name_prefix=file_name_prefix,
                epoch=epoch+1,
                **logs)
        else:
            file_name = '{file_name_prefix}{epoch:02d}.h5'.format(
                file_name_prefix=file_name_prefix,
                epoch=epoch+1)
        os.path.isdir(self._checkpoint_dir) or os.makedirs(self._checkpoint_dir)
        file_path = os.path.join(self._checkpoint_dir, file_name)
        self._model_nima.save_weights(file_path)

    @staticmethod
    def accuracy(y_true, y_pred, threshold=0.5):
        '''Compute classification accuracy with a fixed threshold on distances.

        Args:
            y_true (np.ndarray): an array of ground truth labels, known as `y`.
            y_pred (np.ndarray): an array of predict labels, known as `y_hat`

        Returns:
            Result of the classification accuracy.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < threshold, y_true.dtype)))

    @staticmethod
    def nima_mean_score(scores):
        '''Evaluate the mean score of the predict results from NIMA model.

        Args:
            scores (np.array): an array of results from `predict()`.

        Returns:
            mean (np.array): an array of mean scores.
        '''
        si = np.arange(1., 11.)
        mean = np.sum(scores * si, axis=1)
        return mean

    @staticmethod
    def nima_std_score(scores):
        '''Evaluate the std score of the predict results from NIMA model.

        Args:
            scores (np.array): an array of results from `predict()`.

        Returns:
            std (np.array): an array of std scores.
        '''
        si = np.arange(1., 11.)
        mean = np.sum(scores * si, axis=1).reshape(-1, 1)
        std = np.sqrt(np.sum((si - mean)**2 * scores, axis=1))
        return std

    def load_data(self, image_dir, data_path, sep=' ', columns=['file_name', 'label']):
        '''Load dataset with `.csv` data file.
        If data file is used to train or test model,it should contain two columns.
            Eg. `['file_name', 'label']`.
        Otherwise, data file is used to predict model, it should contain one column.
            Eg. `['file_name']`.

        Args:
            image_dir (str): the image samples directory.
            data_path (str): the data file path.
                Data file should exactly match the `columns` argument.
            sep (str, optional): the data file separator.
                Defaults to ' '.
            columns (list, optional): the data header columns.
                Defaults to ['file_name', 'label'].

        Returns:
            data (tuple): a tuple of data results.
                If data file has two columns, it returns a tuple of (x, y) to train or test.
                Otherwise, data file should have only one column and it returns a tuple (x, ) to predict.
        '''
        assert os.path.isdir(image_dir), 'Error: Invalid image dir `{}`'.format(image_dir)
        assert os.path.isfile(data_path), 'Error: Invalid data path `{}`'.format(data_path)
        assert 1 <= len(columns) <= 2, ('Error: Invalid columns, two columns for training and testing,'
                                        ' one for predicting')

        if len(columns) == 2:
            x_raw, y_raw = [], []
            data_source = pd.read_csv(data_path, sep=sep, usecols=columns)
            x = data_source[columns[0]].apply(lambda x: os.path.join(image_dir, x))
            y = data_source[columns[1]]
            return (x.values, y.values)

        else:
            x_raw = []
            data_source = pd.read_csv(data_path, sep=sep, usecols=columns)
            x = data_source[columns[0]].apply(lambda x: os.path.join(image_dir, x))
            return (x.values,)

    def build(self,
              nima_weight_path=None):
        '''Build NIMA network only.

        Args:
            nima_weight_path (str, optional): the file path of NIMA network weight.
                Defaults to None.

        Returns:
            model (keras.Model): the NIMA network in keras.
        '''
        self._model_nima = self._build_nima_network(weight_path=nima_weight_path)
        return self._model_nima

    def train(self,
              train_raw,
              val_raw=None,
              num_classes=10,
              nima_weight_path=None,
              epochs=20,
              batch_size=64,
              optimizer=None,
              callback=None,
              **kwargs):
        '''Train Siamese NIMA netrowk.

        Args:
            train_raw (tuple): a tuple of train raw data, loaded by `load_data()`.
            val_raw (tuple, optional): a tuple of validate raw data, loaded by `load_data()`.
                Defaults to None.
            num_classes (int, optional): the number of classes that dataset have.
                Defaults to 10.
            nima_weight_path (str, optional): the file path of NIMA network weight.
                If None, no network weight will be loaded, known as retrain.
                Otherwise, NIMA weight will be loaded, known as fine-tuning.
                Defaults to None.
            epochs (int, optional): the number of epoches to train model.
                Defaults to 20.
            batch_size (int, optional): the size of a batch to train model.
                Defaults to 64.
            optimizer (object, optional): the optimizer to train model.
                If None, `Adam(lr=1e-3)` will be used to train model.
                Defaults to None.
            callback (object | list, optional): the callback instance or a list of callback instances.
                See https://keras.io/callbacks/ for more details.
                Defaults to None, callback model checkpoints ONLY.
            **kwargs:
                Other keyword arguments to train model.
                See https://keras.io/models/model/#fit_generator for more details.
        '''
        # load data
        train_x_raw, train_y_raw = train_raw
        train_gen = DataSequence(train_x_raw, train_y_raw,
                                 batch_size=batch_size, num_classes=num_classes)
        val_gen = None
        if val_raw and len(val_raw) == 2:
            val_x_raw, val_y_raw = val_raw
            val_gen = DataSequence(val_x_raw, val_y_raw,
                                   batch_size=batch_size, num_classes=num_classes)

        # define network
        self._model_nima = self._build_nima_network(weight_path=nima_weight_path)
        self._model_siamese = self._build_siamese_network(self._model_nima)

        # callback
        checkpoint = LambdaCallback(on_epoch_end=self._checkpoint_on_epoch_end)
        callbacks = [checkpoint]
        if isinstance(callback, list):
            callbacks += callback
        elif callback is not None:
            callbacks += [callback]

        # train
        optimizer = optimizer or Adam(lr=1e-3)
        self._model_siamese.compile(optimizer=optimizer,
                                    loss=self._contrastive_loss,
                                    metrics=[self.accuracy])
        kwargs.setdefault('verbose', 1)
        kwargs.setdefault('epochs', epochs)
        self._model_siamese.fit_generator(train_gen,
                                          validation_data=val_gen,
                                          callbacks=callbacks,
                                          **kwargs)

    def predict(self,
                predict_raw,
                batch_size=1,
                target_size=None,
                nima_weight_path=None,
                to_df=True,
                **kwargs):
        '''Predict with NIMA network.
        Two arguments `batch_size` and `target_size` will work together.
        If `target_size==None`, then `batch_size` will be set as 1 forcely,
            in this way each image will be loaded and predicted with the original size.
        Otherwise, a batch of images will be loaded and resized by a fixed `target_size`.

        Args:
            predict_raw (tuple): a tuple of train raw data, loaded by `load_data()`.
            batch_size (int, optional): the size of a batch to train model.
                Defaults to 1.
            target_size (tuple, optional): the size of an image that will be resized.
                Defaults to None.
            nima_weight_path (str, optional): the file path of NIMA network weight.
                If None, the model will be loaded from the current instance after training.
                Defaults to None.
            to_df (bool, optional): whether the results are returned as a dataframe.
                Defaults to True.

        Returns:
            results (np.ndarray | pd.Dataframe): the predicted results.
        '''
        assert self._model_nima or nima_weight_path, 'Error: Invalid nima_weight_path {}'.format(
                                                     nima_weight_path)

        # load data
        if isinstance(predict_raw, tuple):
            predict_x_raw, *_ = predict_raw
        else:
            predict_x_raw = predict_raw

        if target_size is None and batch_size != 1:
            print('Warning: force `batch_size=1` due to `target_size==None`')
            batch_size = 1
        predict_gen = DataSequencePred(predict_x_raw,
                                       batch_size=batch_size,
                                       target_size=target_size)

        # define network
        if nima_weight_path:
            self._model_nima = self._build_nima_network(weight_path=nima_weight_path,
                                                        layer_to_freeze=None)

        # predict
        kwargs.setdefault('verbose', 1)
        results = self._model_nima.predict_generator(predict_gen, **kwargs)

        if to_df == False:
            return results

        data_dict = {
            'source': predict_x_raw,
            'mean': self.nima_mean_score(results),
            'std': self.nima_std_score(results),
            'scores': list(results),
        }
        results_df = pd.DataFrame(data_dict)
        return results_df
