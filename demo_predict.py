'''Predict with NIMA model network.'''

from model.siamese_nima import SiameseNIMA


if __name__ == '__main__':
    # dirs and paths to load data
    predict_image_dir = './assets/demo/predict_images'
    predict_data_path = './assets/demo/predict_data.csv'

    # load data and train model
    siamese = SiameseNIMA()
    predict_raw = siamese.load_data(predict_image_dir, predict_data_path,
                                    columns=['file_name'])
    results = siamese.predict(predict_raw,
                              nima_weight_path='./assets/weights/nima_weights_pre_trained.h5')
    print(results)
