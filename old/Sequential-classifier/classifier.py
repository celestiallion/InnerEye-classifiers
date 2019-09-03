from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.optimizers import SGD
import os
import h5py
import sys
sys.path.append('.')
from networks import get_model
from data_utils import Train_data_generator
from data_utils import Valid_data_generator


batch_size = 64
epochs = 50
num_classes = 7
loss = 'categorical_crossentropy'
optimizer = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
metrics = ['accuracy']
save_dir = './trained_models/'
os.makedirs(save_dir, exist_ok=True)
model_name = 'Model_InnerEye-classifier-Sequential-64x64-{epoch:02d}-{val_acc:.4f}.h5'


model = get_model()
print(model.summary())
# '''
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

file_path = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
csv_logger = CSVLogger(os.path.join(save_dir, 'InnerEye-classifier-Sequential-64x64.log'), separator=',', append=False)
train_data_generator = Train_data_generator(batch_size)
valid_data_generator = Valid_data_generator(batch_size)

model.fit_generator(generator=train_data_generator, steps_per_epoch=int(171312/batch_size), epochs=epochs, verbose=True, callbacks=[checkpoint, reduce_lr, csv_logger], validation_data=valid_data_generator, validation_steps=int(35374/batch_size), workers=1, use_multiprocessing=False, shuffle=True)
# '''
