#!/home/adnan/anaconda3/envs/tf1.14/bin/python
from tensorflow.python.keras.models import load_model
from networks import get_model, num_classes
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.models import Model
import numpy as np
import h5py
from PIL import Image
import argparse

model_weight_file = '/home/adnan/Downloads/Model_V1-141-0.7997.h5'

loss = { 'reconstruction': losses.mean_squared_error, 'predictions': losses.categorical_crossentropy }
optimizer = SGD(lr=0.005, momentum=0.9, decay=1e-6, nesterov=True)
metrics = { 'predictions': [categorical_accuracy] }
class_weight = { 'reconstruction': 0.05, 'predictions': 1. }

model = get_model()
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.load_weights(model_weight_file)


def _get_output(r):
    unfiltered, filtered = r[0] * (29/30), np.sum(r[1:]) * (1/30)
    x = unfiltered + filtered
    return unfiltered / x, filtered / x
    # x = np.sum(r[1:]) / 30  # class reweighting is done here.
    # return 1.-x, x  # unfiltered and filtered


def _get_image_test_result(image_file_path):
    img = np.asarray(Image.open(image_file_path).resize((64, 64), Image.LANCZOS)).reshape(1, 64, 64, 3)

    if np.array_equal(img[:, :, :, 0], img[:, :, :, 1]) and np.array_equal(img[:, :, :, 1], img[:, :, :, 2]):
        return 0., 1.

    img = img.astype('float32') / 255
    result = model.predict([img])

    r = np.array([x for x in result[1].tolist()[0]] + [0.])
    return _get_output(r)


def main(parser_args):
    prediction = _get_image_test_result(parser_args.path)
    print('Unfiltered with a confidence of: ' + str(prediction[0] * 100))
    print('Filtered with a confidence of: ' + str(prediction[1] * 100))
    # print('Unfiltered with a confidence of: ' + str(prediction[0] * 100))
    # print('Filtered with a confidence of: ' + str(prediction[1] * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/home/adnan/Desktop/26.jpg', type=str, help='Path to the image file.')
    parser_args = parser.parse_args()

    main(parser_args)
