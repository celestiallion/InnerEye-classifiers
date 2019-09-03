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

model_weight_file = '/home/adnan/InnerEye-Machine-Learning/Model_V1-175-0.8070.h5'
loss = { 'reconstruction': losses.mean_squared_error, 'predictions': losses.categorical_crossentropy }
optimizer = SGD(lr=0.005, momentum=0.9, decay=1e-6, nesterov=True)
metrics = { 'predictions': [categorical_accuracy] }
class_weight = { 'reconstruction': 0.05, 'predictions': 1. }

model = get_model()
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.load_weights(model_weight_file)

hf = h5py.File('/home/adnan/Datasets/New-InnerEye-dataset/new-innereye-dataset-multi-labeled-64x64.h5', 'r')
X_test, y_test = hf['X_test'][:], hf['y_test'][:]

count = 0

for c, (img, y) in enumerate(zip(X_test, y_test)):
    img = img.astype('float32')/255
    img = img.reshape((1, 64, 64, 3))
    prediction = np.argmax(model.predict([img])[1])
    if prediction == y:
        count += 1
    print(c)

print(count/X_test.shape[0])
