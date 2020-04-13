'''Train Siamese NIMA model networks.'''

from model.siamese_nima import SiameseNIMA


if __name__ == '__main__':
    # dirs and paths to load data
    train_image_dir = './assets/demo/train_images'
    train_data_path = './assets/demo/train_data.csv'

    # load data and train model
    siamese = SiameseNIMA(output_dir='./assets')
    train_raw = siamese.load_data(train_image_dir, train_data_path)
    siamese.train(train_raw,
                  epochs=5,
                  batch_size=16,
                  nima_weight_path='./assets/weights/nima_weights_pre_trained.h5')
