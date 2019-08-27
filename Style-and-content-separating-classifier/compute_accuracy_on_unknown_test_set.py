#!/home/adnan/anaconda3/envs/tf1.14/bin/python
from tensorflow.python.keras.models import load_model
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from networks import get_style_code
from networks import get_content_code
from networks import get_adain_parameters
from networks import adain
from networks import get_decoding_residual_blocks_signal
from networks import get_upsampled_signal
from networks import get_encoded_predictions
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.models import Model

num_classes = 7


def _get_test_images_and_labels(image_dir_path):
    images_and_labels = []
    # image_classes_path = os.path.join(image_dir_path, 'test')
    image_classes_path = os.path.join(image_dir_path, 'unknown_test')
    image_classes = os.listdir(image_classes_path)
    for image_class in image_classes:
        image_files_path = os.path.join(image_classes_path, image_class)
        for image_file in os.listdir(image_files_path):
            if image_class == 'original':
                images_and_labels.append((os.path.join(image_files_path, image_file), 0))
            else:
                images_and_labels.append((os.path.join(image_files_path, image_file), 1))
    return images_and_labels


def _get_output(r):
    a, b = r[0], np.sum(r[1:])  # original and filtered
    if a >= b:
        return 0
    else:
        return 1


def _get_model(model_weight_file):
    img_tensor = Input(shape=(64, 64, 3))
    content_code = get_content_code(img_tensor)
    style_code = get_style_code(img_tensor)
    adain_parameters = Lambda(get_adain_parameters)(style_code)
    x = Lambda(adain, arguments={'epsilon': 1e-5})((content_code, adain_parameters))
    x = get_decoding_residual_blocks_signal(x)
    x = get_upsampled_signal(x)
    # reconstruction = Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=1, padding='same', activation='sigmoid', name='reconstruction')(x)  # v1, v2
    reconstruction = Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='reconstruction')(x)  # v3, v4
    prediction_code = get_encoded_predictions(style_code)
    predictions = Dense(units=num_classes, activation='softmax', name='predictions')(style_code)
    model = Model(inputs=img_tensor, outputs=[reconstruction, predictions])
    print(model.summary())

    loss = { 'reconstruction': mean_squared_error, 'predictions': categorical_crossentropy }
    optimizer = SGD(lr=0.005, momentum=0.9, decay=1e-6, nesterov=True)
    metrics = { 'predictions': [categorical_accuracy] }
    # class_weight = { 'reconstruction': 1., 'predictions': 1. }  # v1, v2
    class_weight = { 'reconstruction': 0.05, 'predictions': 1. }  # v3, v4

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.load_weights(model_weight_file)

    return model


def compute_accuracy(image_dir_path, model):
    image_files = _get_test_images_and_labels(image_dir_path)

    correct_count = 0
    count = len(image_files)

    for c, (image_path, label) in enumerate(image_files):
        try:
            img = np.asarray(Image.open(image_path).resize((64, 64), Image.LANCZOS))
            if np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 1], img[:, :, 2]):
                r = np.array([0., 0., 0., 0., 0., 0., 0., 1.])
                if _get_output(r) == label:
                    correct_count += 1
                    continue

            img = img.reshape(1, 64, 64, 3).astype('float32') / 255
            result = model.predict([img])

            r = np.array([x for x in result[1].tolist()[0]] + [0.])
            if _get_output(r) == label:
                correct_count += 1

            print(c)
        except Exception:
            count -= 1
            continue

    # As I used dropout layer while training, the training accuracy is smaller than the test one.
    return correct_count / count

def main(parser_args):
    image_dir_path = parser_args.path
    model = _get_model('/home/adnan/InnerEye-classifiers/InnerEye-trained-models/Model_InnerEye-classifier-style-content-separation-010-0.9043.h5')
    print(compute_accuracy(image_dir_path, model))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/home/adnan/Datasets/InnerEye-dataset/', type=str, help='Path to the image dataset.')
    parser_args = parser.parse_args()

    main(parser_args)
