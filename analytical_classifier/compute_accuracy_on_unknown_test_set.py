#!/home/adnan/anaconda3/envs/tf1.14/bin/python
from tensorflow.python.keras.models import load_model
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from networks import get_model, num_classes
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.models import Model


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
    idx = np.argmax(r)
    return 0 if idx == 0 else 1


def _get_model(model_weight_file):
    loss = { 'reconstruction': losses.mean_squared_error, 'predictions': losses.categorical_crossentropy }
    optimizer = SGD(lr=0.005, momentum=0.9, decay=1e-6, nesterov=True)
    metrics = { 'predictions': [categorical_accuracy] }
    class_weight = { 'reconstruction': 0.05, 'predictions': 1. }

    model = get_model()
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
    model = _get_model('/home/adnan/analytical-classifier-for-InnerEye/base/trained-models/Model_analytical-classifier-for-InnerEye-01-0.8579.h5')
    print(compute_accuracy(image_dir_path, model))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/home/adnan/Datasets/InnerEye-dataset/', type=str, help='Path to the image dataset.')
    parser_args = parser.parse_args()

    main(parser_args)
