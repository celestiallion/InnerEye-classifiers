import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import add
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
import sys
sys.path.append('.')
from custom_layers import InstanceNormalization


def get_encoding_residual_blocks_signal(x):
    y = Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding='same', name='encode_residual_conv_1')(x)
    y = BatchNormalization(name='encode_residual_bn_1')(y)  # v1, v2, v3
    y = LeakyReLU(name='encode_residual_relu_1')(y)
    y = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', name='encode_residual_conv_2')(y)
    y = BatchNormalization(name='encode_residual_bn_2')(y)  # v1, v2, v3
    y = LeakyReLU(name='encode_residual_relu_2')(y)
    y = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', name='encode_residual_conv_3')(y)
    y = BatchNormalization(name='encode_residual_bn_3')(y)  # v1, v2, v3
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', name='encode_residual_conv_4')(x)
    x = BatchNormalization(name='encode_residual_bn_4')(x)  # v1, v2, v3
    y = add([x, y], name='encode_residual_add_1')
    x = LeakyReLU(name='encode_residual_relu_3')(y)
    y = Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding='same', name='encode_residual_conv_5')(x)
    y = BatchNormalization(name='encode_residual_bn_5')(y)  # v1, v2, v3
    y = LeakyReLU(name='encode_residual_relu_4')(y)
    y = Conv2D(filters=32, kernel_size=(3, 3), strides=1,  padding='same', name='encode_residual_conv_6')(y)
    y = BatchNormalization(name='encode_residual_bn_6')(y)  # v1, v2, v3
    y = LeakyReLU(name='encode_residual_relu_5')(y)
    y = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', name='encode_residual_conv_7')(y)
    y = BatchNormalization(name='encode_residual_bn_7')(y)  # v1, v2, v3
    y = add([x, y], name='encode_residual_add_2')
    x = LeakyReLU(name='encode_residual_relu_6')(y)
    return x


def get_downsampled_signal(img_tensor, module_name):
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=2, padding='same', name='downsample_{0}_conv_1'.format(module_name))(img_tensor)
    x = InstanceNormalization(name='downsample_{0}_in_1'.format(module_name))(x) if module_name == 'content' else x
    x = LeakyReLU(name='downsample_{0}_relu_1'.format(module_name))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='same', name='downsample_{0}_conv_2'.format(module_name))(x)
    x = InstanceNormalization(name='downsample_{0}_in_2'.format(module_name))(x) if module_name == 'content' else x
    x = LeakyReLU(name='downsample_{0}_relu_2'.format(module_name))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', name='downsample_{0}_conv_3'.format(module_name))(x)
    x = InstanceNormalization(name='downsample_{0}_in_3'.format(module_name))(x) if module_name == 'content' else x
    x = LeakyReLU(name='downsample_{0}_relu_3'.format(module_name))(x)
    return x


def get_content_code(img_tensor):
    x = get_downsampled_signal(img_tensor, 'content')
    x = get_encoding_residual_blocks_signal(x)
    return x


def get_style_code(img_tensor):
    x = get_downsampled_signal(img_tensor, 'style')
    x = GlobalAveragePooling2D(name='encode_style_global_avgpool')(x)
    x = Dense(units=64, activation='relu', name='encode_style_fc')(x)
    return x


def get_adain_parameters(x):
    mean = Dense(units=64, activation='relu', name='adain_parameters_mean')(x)
    mean = tf.reshape(mean, shape=[-1, 1, 1, 64])
    variance = Dense(units=64, activation='relu', name='adain_parameters_variance')(x)
    variance = tf.reshape(variance, shape=[-1, 1, 1, 64])
    return mean, variance


def adain(xargs, epsilon=1e-5):
    content, style = xargs[0], xargs[1]
    content_mean, content_variance = tf.nn.moments(content, axes=[1, 2], keepdims=True)
    style_mean, style_variance = style

    content_variance_adjusted = tf.sqrt(tf.add(content_variance, epsilon), name='content_variance_adjusted')
    style_variance_adjusted = tf.sqrt(tf.add(style_variance, epsilon), name='style_variance_adjusted')

    return (content - content_mean) * style_variance_adjusted / content_variance_adjusted + style_mean


def get_decoding_residual_blocks_signal(x):
    y = Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding='same', name='decode_residual_conv_1')(x)
    y = BatchNormalization(name='decode_residual_bn_1')(y)  # v1, v2, v3
    y = LeakyReLU(name='decode_residual_relu_1')(y)
    y = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', name='decode_residual_conv_2')(y)
    y = BatchNormalization(name='decode_residual_bn_2')(y)  # v1, v2, v3
    y = LeakyReLU(name='decode_residual_relu_2')(y)
    y = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', name='decode_residual_conv_3')(y)
    y = BatchNormalization(name='decode_residual_bn_3')(y)  # v1, v2, v3
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', name='decode_residual_conv_4')(x)
    x = BatchNormalization(name='decode_residual_bn_4')(x)  # v1, v2, v3
    y = add([x, y], name='decode_residual_add_1')
    x = LeakyReLU(name='decode_residual_relu_3')(y)
    y = Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding='same', name='decode_residual_conv_5')(x)
    y = BatchNormalization(name='decode_residual_bn_5')(y)  # v1, v2, v3
    y = LeakyReLU(name='decode_residual_relu_4')(y)
    y = Conv2D(filters=32, kernel_size=(3, 3), strides=1,  padding='same', name='decode_residual_conv_6')(y)
    y = BatchNormalization(name='decode_residual_bn_6')(y)  # v1, v2, v3
    y = LeakyReLU(name='decode_residual_relu_5')(y)
    y = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', name='decode_residual_conv_7')(y)
    y = BatchNormalization(name='decode_residual_bn_7')(y)  # v1, v2, v3
    y = add([x, y], name='decode_residual_add_2')
    x = LeakyReLU(name='decode_residual_relu_6')(y)
    return x


def get_upsampled_signal(x):
    y = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=1, padding='same', name='decode_convtrans_1')(x)
    y = LeakyReLU(name='decode_relu_1')(y)
    y = UpSampling2D(size=(2, 2), name='decode_upsample_1')(y)
    y = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=1, padding='same', name='decode_convtrans_2')(x)
    y = LeakyReLU(name='decode_relu_2')(y)
    y = UpSampling2D(size=(2, 2), name='decode_upsample_2')(y)
    y = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=1, padding='same', name='decode_convtrans_3')(y)
    y = LeakyReLU(name='decode_relu_3')(y)
    y = UpSampling2D(size=(2, 2), name='decode_upsample_3')(y)
    y = Conv2DTranspose(filters=16, kernel_size=(2, 2), strides=1, padding='same', name='decode_convtrans_4')(y)
    y = LeakyReLU(name='decode_relu_4')(y)
    return y


def get_encoded_predictions(input_code):
    return Dropout(0.125, name='encoded_predictions_dropout')(input_code)
