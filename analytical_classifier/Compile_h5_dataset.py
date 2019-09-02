import os
from PIL import Image
import numpy as np
from random import sample
from random import shuffle
import h5py
import gc
# from preprocessing_utils import get_histogram_matched_image

IMAGE_ROOT = '/home/adnan/Datasets/New-InnerEye-dataset'
FILTERS = ['_1977', 'aden', 'brannan', 'brooklyn', 'clarendon', 'earlybird', 'gingham', 'gotham', 'hudson', 'inkwell', 'kelvin', 'lark', 'lofi', 'lomo', 'maven', 'mayfair', 'moon', 'nashville', 'perpetua', 'reyes', 'rise', 'sepia', 'slumber', 'stinson', 'toaster', 'valencia', 'walden', 'willow', 'xpro2']
filter_labels = {'_1977': 1, 'aden': 2, 'brannan': 3, 'brooklyn': 4, 'clarendon': 5, 'earlybird': 6, 'gingham': 7, 'gotham': 8, 'hudson': 9, 'inkwell': 10, 'kelvin': 11, 'lark': 12, 'lofi': 13, 'lomo': 14, 'maven': 15, 'mayfair': 16, 'moon': 17, 'nashville': 18, 'perpetua': 19, 'reyes': 20, 'rise': 21, 'sepia': 22, 'slumber': 23, 'stinson': 24, 'toaster': 25, 'valencia': 26, 'walden': 27, 'willow': 28, 'xpro2': 29}

X_train, X_valid, X_test, y_train, y_valid, y_test = [], [], [], [], [], []

with h5py.File(os.path.join(IMAGE_ROOT, 'new-innereye-dataset-multi-labeled-64x64.h5'), 'w') as hf:
    # Add examples from training dataset
    dir_path = os.path.join(IMAGE_ROOT, 'train', 'original')
    dir_files = os.listdir(dir_path)
    for c, file_name in enumerate(dir_files):
        img = np.asarray(Image.open(os.path.join(dir_path, file_name)).resize((64, 64), Image.LANCZOS))
        # img = get_histogram_matched_image(img)
        if img.shape != (64, 64, 3):
            continue
        X_train.append(img)
        y_train.append(0)
        print('original: ' + str(c))

    for filter in FILTERS:
        # Add examples from training dataset.
        dir_path = os.path.join(IMAGE_ROOT, 'train', filter)
        dir_files = os.listdir(dir_path)
        for c, file_name in enumerate(dir_files):
            img = np.asarray(Image.open(os.path.join(dir_path, file_name)).resize((64, 64), Image.LANCZOS))
            # img = get_histogram_matched_image(img)
            if img.shape != (64, 64, 3):
                continue
            X_train.append(img)
            y_train.append(filter_labels[filter])
            print(filter + ' ' + str(c))

    placeholder = list(zip(X_train, y_train))
    shuffle(placeholder)
    X_train, y_train = zip(*placeholder)
    placeholder = None

    hf.create_dataset('X_train', data=np.array(X_train))
    hf.create_dataset('y_train', data=np.array(y_train))

    del X_train
    del y_train
    gc.collect()

    #Add examples from validation dataset
    dir_path = os.path.join(IMAGE_ROOT, 'valid', 'original')
    dir_files = os.listdir(dir_path)
    for c, file_name in enumerate(dir_files):
        img = np.asarray(Image.open(os.path.join(dir_path, file_name)).resize((64, 64), Image.LANCZOS))
        # img = get_histogram_matched_image(img)
        if img.shape != (64, 64, 3):
            continue
        X_valid.append(img)
        y_valid.append(0)
        print('valid:: original: ' + str(c))

    for filter in FILTERS:
        # Add examples from validation dataset.
        dir_path = os.path.join(IMAGE_ROOT, 'valid', filter)
        dir_files = os.listdir(dir_path)
        for c, file_name in enumerate(dir_files):
            img = np.asarray(Image.open(os.path.join(dir_path, file_name)).resize((64, 64), Image.LANCZOS))
            # img = get_histogram_matched_image(img)
            if img.shape != (64, 64, 3):
                continue
            X_valid.append(img)
            y_valid.append(filter_labels[filter])
            print('valid:: ' + filter + ' ' + str(c))

    placeholder = list(zip(X_valid, y_valid))
    shuffle(placeholder)
    X_valid, y_valid = zip(*placeholder)
    placeholder = None

    hf.create_dataset('X_valid', data=np.array(X_valid))
    hf.create_dataset('y_valid', data=np.array(y_valid))

    del X_valid
    del y_valid
    gc.collect()

    dir_path = os.path.join(IMAGE_ROOT, 'test', 'original')
    dir_files = os.listdir(dir_path)
    for c, file_name in enumerate(dir_files):
        img = np.asarray(Image.open(os.path.join(dir_path, file_name)).resize((64, 64), Image.LANCZOS))
        # img = get_histogram_matched_image(img)
        if img.shape != (64, 64, 3):
          continue
        X_test.append(img)
        y_test.append(0)
        print('test:: original: ' + str(c))

    for filter in FILTERS:
        # Add examples from test dataset.
        dir_path = os.path.join(IMAGE_ROOT, 'test', filter)
        dir_files = os.listdir(dir_path)
        for c, file_name in enumerate(dir_files):
            img = np.asarray(Image.open(os.path.join(dir_path, file_name)).resize((64, 64), Image.LANCZOS))
            # img = get_histogram_matched_image(img)
            if img.shape != (64, 64, 3):
              continue
            X_test.append(img)
            y_test.append(filter_labels[filter])
            print('test:: ' + filter + ' ' + str(c))

    placeholder = list(zip(X_test, y_test))
    shuffle(placeholder)
    X_test, y_test = zip(*placeholder)
    placeholder = None

    hf.create_dataset('X_test', data=np.array(X_test))
    hf.create_dataset('y_test', data=np.array(y_test))

    del X_test
    del y_test
    gc.collect()

    print('Done with compiling the dataset!')
    hf.close()
