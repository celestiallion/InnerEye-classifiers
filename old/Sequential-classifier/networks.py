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

num_classes = 7


def get_model():
    img_tensor = Input(shape=(64, 64, 3))
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, padding='same')(img_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = AvgPool2D(pool_size=(2, 2), strides=1, padding='valid')(x)  # Blurred output produced by AvgPool2D, intuitively, gives a better estimate of filters used rather than sharp one produced by MaxPool2D because in blur output the neighboring colors are aggregated and sharp outputs often contain max values due to presence of edges.
    x = Flatten()(x)
    x = Dense(units=32, activation='relu')(x)
    x = Dropout(0.25)(x)
    predicted_class = Dense(units=num_classes, activation='softmax')(x)

    model = Model(inputs=[img_tensor], outputs=[predicted_class])

    return model
