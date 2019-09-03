from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.optimizers import SGD
import os
import sys
sys.path.append('.')
from networks import get_style_code
from networks import get_content_code
from networks import get_adain_parameters
from networks import adain
from networks import get_decoding_residual_blocks_signal
from networks import get_upsampled_signal
from networks import get_encoded_predictions
from data_utils import Train_data_generator
from data_utils import Valid_data_generator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 32  # We want to avoid too much generalization over the style space to preserve style information as evidenced by the principle of Instance Normalization; ref: Ulyanov et al. 2016
num_classes = 7
epochs = 200
save_dir = './InnerEye-trained-models'
os.makedirs(save_dir, exist_ok=True)
model_name = 'Model_InnerEye-classifier-style-content-separation-{epoch:03d}-{val_predictions_categorical_accuracy:.4f}.h5'


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
# '''
loss = { 'reconstruction': mean_squared_error, 'predictions': categorical_crossentropy }
optimizer = SGD(lr=0.005, momentum=0.9, decay=1e-6, nesterov=True)
metrics = { 'predictions': [categorical_accuracy] }
# class_weight = { 'reconstruction': 1., 'predictions': 1. }  # v1, v2
class_weight = { 'reconstruction': 0.05, 'predictions': 1. }  # v3, v4

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
# '''
# '''
file_path = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(file_path, monitor='val_predictions_categorical_accuracy', verbose=1, save_best_only=True, mode='auto', save_weights_only=True, period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_predictions_loss', factor=0.5, patience=25, verbose=1, mode='auto', min_delta=1e-6, cooldown=0, min_lr=0)
csv_logger = CSVLogger(os.path.join(save_dir, 'InnerEye-classifier-style-content-separation.log'), separator=',', append=False)
train_data_generator = Train_data_generator(batch_size)
valid_data_generator = Valid_data_generator(batch_size)

model.fit_generator(generator=train_data_generator, steps_per_epoch=int(171312/batch_size), epochs=epochs, verbose=1, callbacks=[checkpoint, reduce_lr, csv_logger], validation_data=valid_data_generator, validation_steps=int(35374/batch_size), workers=1, class_weight=class_weight, use_multiprocessing=False, shuffle=True)
# '''
