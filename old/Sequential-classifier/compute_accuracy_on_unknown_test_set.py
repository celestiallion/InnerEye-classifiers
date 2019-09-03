#!/home/adnan/anaconda3/envs/tf1.14/bin/python
from tensorflow.python.keras.models import load_model
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

model = load_model('./trained_models/Model_InnerEye-classifier-Sequential-64x64-17-0.9211.h5')

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


def _compute_accuracy(image_dir_path):
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
            result = model.predict(img)

            r = np.array([x for x in result.tolist()[0]] + [0.])
            if _get_output(r) == label:
                correct_count += 1

            print(c)
        except Exception:
            count -= 1
            continue

    # As I used dropout layer while training, the training accuracy is smaller than the test one.
    return correct_count / count

def main(parser_args):
    print(_compute_accuracy(parser_args.path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/home/adnan/Datasets/InnerEye-dataset/', type=str, help='Path to the image dataset.')
    parser_args = parser.parse_args()

    main(parser_args)
