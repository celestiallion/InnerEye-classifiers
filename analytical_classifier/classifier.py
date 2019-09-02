from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.optimizers import SGD
import os
import sys
sys.path.append('.')
from networks import get_model, num_classes
from data_utils import Train_data_generator, Valid_data_generator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# We want to avoid too much generalization over the style space to preserve style information
# as evidenced by the principle of Instance Normalization; ref: Ulyanov et al. 2016
classifier_id = 'V1'
batch_size = 32
epochs = 50
save_dir = '/home/adnan/InnerEye-Machine-Learning/captain-buet/trained-models'
os.makedirs(save_dir, exist_ok=True)
model_name = 'Model_captain-buet-V1-{epoch:02d}-{val_predictions_categorical_accuracy:.4f}.h5'

loss = { 'reconstruction': losses.mean_squared_error, 'predictions': losses.categorical_crossentropy }
optimizer = SGD(lr=0.005, momentum=0.9, decay=1e-6, nesterov=True)
metrics = { 'predictions': [metrics.categorical_accuracy] }
class_weight = { 'reconstruction': 0.05, 'predictions': 1. }

model = get_model()
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print(model.summary())

# '''
file_path = os.path.join(save_dir, model_name)
checkpoint = callbacks.ModelCheckpoint(file_path, monitor='val_predictions_categorical_accuracy', verbose=1, save_best_only=True, mode='auto', save_weights_only=True, period=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_predictions_loss', factor=0.25, patience=10, verbose=1, mode='auto', min_delta=1e-6, cooldown=0, min_lr=0)
csv_logger = callbacks.CSVLogger(os.path.join(save_dir, 'Log_captain-buet-V1.log'), separator=',', append=False)
train_data_generator = Train_data_generator(batch_size)
valid_data_generator = Valid_data_generator(batch_size)

model.fit_generator(generator=train_data_generator, steps_per_epoch=int(210030/batch_size), epochs=epochs, verbose=1, callbacks=[checkpoint, reduce_lr, csv_logger], validation_data=valid_data_generator, validation_steps=int(7530/batch_size), workers=1, class_weight=class_weight, use_multiprocessing=False, shuffle=True)
# '''
