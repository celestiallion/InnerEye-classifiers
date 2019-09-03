#!/home/adnan/anaconda3/envs/tf1.14/bin/python
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import AvgPool2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.callbacks import CSVLogger
# from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils  import Sequence
from tensorflow.python.keras.utils import to_categorical
import os
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATASET_ROOT = '/home/adnan/Datasets/New-InnerEye-dataset/'

batch_size = 64
epochs = 50
num_classes = 30
loss = 'categorical_crossentropy'
# optimizer = SGD(lr=0.05, momentum=0.9, decay=1e-6, nesterov=True)
optimizer = Adam(lr=0.0005, amsgrad=True)
metrics = ['accuracy']
save_dir = '/home/adnan/InnerEye-Machine-Learning/trained-models'
model_name = 'Model_Sequential-v1-{epoch:02d}-{val_acc:.4f}.h5'
hf = h5py.File('/home/adnan/Datasets/New-InnerEye-dataset/new-innereye-dataset-multi-labeled-64x64.h5', 'r')


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
        return batch_X, batch_y


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
        return batch_X, batch_y


img_tensor = Input(shape=(64, 64, 3))
x = Conv2D(filters=32, kernel_size=(5, 5), strides=2, padding='same')(img_tensor)
x = LeakyReLU()(x)
x = Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='same')(x)
x = LeakyReLU()(x)
x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(x)
x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(x)
x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
x = LeakyReLU()(x)
x = AvgPool2D(pool_size=(2, 2), strides=1, padding='valid')(x)  # Blurred output produced by AvgPool2D, intuitively, gives a better estimate of filters used rather than sharp one produced by MaxPool2D because in blur output the neighboring colors are aggregated and sharp outputs often contain max values due to presence of edges.
x = Flatten()(x)
x = Dropout(0.25)(x)
predicted_class = Dense(units=num_classes, activation='softmax')(x)

model = Model(inputs=[img_tensor], outputs=[predicted_class])
print(model.summary())
'''
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

file_path = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
csv_logger = CSVLogger(os.path.join(save_dir, 'Log-Sequential-v1.log'), separator=',', append=False)
train_data_generator = Train_data_generator(batch_size)
valid_data_generator = Valid_data_generator(batch_size)

model.fit_generator(generator=train_data_generator, steps_per_epoch=int(210030/batch_size), epochs=epochs, verbose=True, callbacks=[checkpoint, reduce_lr, csv_logger], validation_data=valid_data_generator, validation_steps=int(7530/batch_size), workers=1, use_multiprocessing=False, shuffle=True)
'''
