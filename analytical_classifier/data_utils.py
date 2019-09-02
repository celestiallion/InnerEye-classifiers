from tensorflow.python.keras.utils  import Sequence
from tensorflow.python.keras.utils import to_categorical
import h5py

num_classes = 30

hf = h5py.File('/home/adnan/Datasets/New-InnerEye-dataset/new-innereye-dataset-multi-labeled-64x64.h5', 'r')
# hf = h5py.File('/content/drive/My Drive/InnerEye-data/new-innereye-dataset-multi-labeled-64x64.h5', 'r')


class Train_data_generator(Sequence):

    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def __len__(self):
        return int(hf['X_train'][:].shape[0] / self.batch_size)

    def __getitem__(self, index):
        batch_X = hf['X_train'][index * self.batch_size: (1 + index) * self.batch_size]
        batch_y = hf['y_train'][index * self.batch_size: (1 + index) * self.batch_size]
        batch_X = batch_X.astype('float32')
        batch_X /= 255
        batch_y = to_categorical(batch_y, num_classes)
        return batch_X, {'reconstruction': batch_X, 'predictions': batch_y}


class Valid_data_generator(Sequence):

    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def __len__(self):
        return int(hf['X_valid'][:].shape[0] / self.batch_size)

    def __getitem__(self, index):
        batch_X = hf['X_valid'][index * self.batch_size: (1 + index) * self.batch_size]
        batch_y = hf['y_valid'][index * self.batch_size: (1 + index) * self.batch_size]
        batch_X = batch_X.astype('float32')
        batch_X /= 255
        batch_y = to_categorical(batch_y, num_classes)
        return batch_X, {'reconstruction': batch_X, 'predictions': batch_y}


class Test_data_generator(Sequence):

    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def __len__(self):
        return int(hf['X_test'][:].shape[0] / self.batch_size)

    def __getitem__(self, index):
        batch_X = hf['X_test'][index * self.batch_size: (1 + index) * self.batch_size]
        batch_y = hf['y_test'][index * self.batch_size: (1 + index) * self.batch_size]
        batch_X = batch_X.astype('float32')
        batch_X /= 255
        batch_y = to_categorical(batch_y, num_classes)
        return batch_X, {'reconstruction': batch_X, 'predictions': batch_y}
