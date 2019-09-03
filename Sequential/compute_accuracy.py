from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import to_categorical
import h5py

hf = h5py.File('/home/adnan/Datasets/New-InnerEye-dataset/new-innereye-dataset-multi-labeled-64x64.h5', 'r')
X_test, y_test = hf['X_test'][:].astype('float32')/255, to_categorical(hf['y_test'][:], num_classes=30)

model = load_model('/home/adnan/InnerEye-Machine-Learning/trained-models/Model_Sequential-v1-18-0.8026.h5')

acc = model.evaluate(X_test, y_test)

print(acc)
