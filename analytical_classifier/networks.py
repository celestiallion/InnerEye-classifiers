import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
import sys
sys.path.append('.')
from custom_layers import InstanceNormalization
import numpy as np

num_classes = 30


def _get_downsampled_signal(img_tensor, module_name, n_filters_lst=[32, 64]):
    x = layers.Conv2D(filters=n_filters_lst[0], kernel_size=(5, 5), strides=2, padding='same', name='downsample_{0}_conv_1'.format(module_name))(img_tensor)
    x = InstanceNormalization(name='downsample_{0}_in_1'.format(module_name))(x) if module_name == 'content' else x
    x = layers.LeakyReLU(name='downsample_{0}_relu_1'.format(module_name))(x)
    x = layers.Conv2D(filters=n_filters_lst[1], kernel_size=(3, 3), strides=2, padding='same', name='downsample_{0}_conv_2'.format(module_name))(x)
    x = InstanceNormalization(name='downsample_{0}_in_2'.format(module_name))(x) if module_name == 'content' else x
    x = layers.LeakyReLU(name='downsample_{0}_relu_2'.format(module_name))(x)
    return x


def _get_content_encoding_residual_blocks_signal(x, n_filters=32, expansion_f=2):
    y = layers.Conv2D(filters=n_filters, kernel_size=(1, 1), strides=1, padding='same', name='content_encode_residual_conv_1')(x)
    y = layers.BatchNormalization(name='content_encode_residual_bn_1')(y)
    y = layers.LeakyReLU(name='content_encode_residual_relu_1')(y)
    y = layers.Conv2D(filters=n_filters, kernel_size=(5, 5), strides=1, padding='same', name='content_encode_residual_conv_2')(y)
    y = layers.BatchNormalization(name='content_encode_residual_bn_2')(y)
    y = layers.LeakyReLU(name='content_encode_residual_relu_2')(y)
    y = layers.Conv2D(filters=n_filters * expansion_f, kernel_size=(1, 1), strides=1, padding='same', name='content_encode_residual_conv_3')(y)
    y = layers.BatchNormalization(name='content_encode_residual_bn_3')(y)
    y = layers.LeakyReLU(name='content_encode_residual_relu_3')(y)
    x = layers.Conv2D(filters=n_filters * expansion_f, kernel_size=(1, 1), strides=1, padding='same', name='content_encode_residual_conv_4')(x)
    x = layers.BatchNormalization(name='content_encode_residual_bn_4')(x)
    y = layers.add([x, y], name='content_encode_residual_add_1')
    x = layers.LeakyReLU(name='content_encode_residual_relu_4')(y)
    y = layers.Conv2D(filters=n_filters, kernel_size=(1, 1), strides=1, padding='same', name='content_encode_residual_conv_5')(x)
    y = layers.BatchNormalization(name='content_encode_residual_bn_5')(y)
    y = layers.LeakyReLU(name='content_encode_residual_relu_5')(y)
    y = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), strides=1,  padding='same', name='content_encode_residual_conv_6')(y)
    y = layers.BatchNormalization(name='content_encode_residual_bn_6')(y)
    y = layers.LeakyReLU(name='content_encode_residual_relu_6')(y)
    y = layers.Conv2D(filters=n_filters * expansion_f, kernel_size=(1, 1), strides=1, padding='same', name='content_encode_residual_conv_7')(y)
    y = layers.BatchNormalization(name='content_encode_residual_bn_7')(y)
    y = layers.add([x, y], name='content_encode_residual_add_2')
    x = layers.LeakyReLU(name='content_encode_residual_relu_7')(y)
    return x


def _get_content_code(img_tensor, n_filters=32, expansion_f=2):
    x = _get_downsampled_signal(img_tensor, 'content')
    x = _get_content_encoding_residual_blocks_signal(x, n_filters=n_filters, expansion_f=expansion_f)
    return x


def _get_style_code(img_tensor):
    x = _get_downsampled_signal(img_tensor, 'style')
    x = layers.GlobalAveragePooling2D(name='encode_style_global_avgpool')(x)
    x = layers.Dense(units=256, activation='relu', name='encode_style_fc')(x)
    return x


def _get_adain_parameters(x):
    mean = layers.Dense(units=64, activation='relu', name='adain_parameters_mean')(x)
    mean = tf.reshape(mean, shape=[-1, 1, 1, 64])
    variance = layers.Dense(units=64, activation='relu', name='adain_parameters_variance')(x)
    variance = tf.reshape(variance, shape=[-1, 1, 1, 64])
    return mean, variance


def adain(xargs, epsilon=1e-5):
    content, style = xargs[0], xargs[1]
    content_mean, content_variance = tf.nn.moments(content, axes=[1, 2], keepdims=True)
    style_mean, style_variance = style

    content_variance_adjusted = tf.sqrt(tf.add(content_variance, epsilon), name='content_variance_adjusted')
    style_variance_adjusted = tf.sqrt(tf.add(style_variance, epsilon), name='style_variance_adjusted')

    return (content - content_mean) * style_variance_adjusted / content_variance_adjusted + style_mean


def _do_variational_autoencoding(input_signal, latent_dim=2):
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='same')(input_signal)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    shape_before_flattening = K.int_shape(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=32, activation='relu')(x)
    z_mean = layers.Dense(units=latent_dim)(x)
    z_log_var = layers.Dense(units=latent_dim)(x)
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    z = z_mean + K.exp(z_log_var) * epsilon
    x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(z)
    x = layers.Reshape(shape_before_flattening[1:])(x)
    x = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(x)
    return x


def _get_decoding_residual_blocks_signal(x, n_filters=32, expansion_f=2):
    y = layers.Conv2D(filters=n_filters, kernel_size=(1, 1), strides=1, padding='same', name='decode_residual_conv_1')(x)
    y = layers.BatchNormalization(name='decode_residual_bn_1')(y)
    y = layers.LeakyReLU(name='decode_residual_relu_1')(y)
    y = layers.Conv2D(filters=n_filters, kernel_size=(5, 5), strides=1, padding='same', name='decode_residual_conv_2')(y)
    y = layers.BatchNormalization(name='decode_residual_bn_2')(y)
    y = layers.LeakyReLU(name='decode_residual_relu_2')(y)
    y = layers.Conv2D(filters=n_filters * expansion_f, kernel_size=(1, 1), strides=1, padding='same', name='decode_residual_conv_3')(y)
    y = layers.BatchNormalization(name='decode_residual_bn_3')(y)
    y = layers.LeakyReLU(name='decode_residual_relu_3')(y)
    x = layers.Conv2D(filters=n_filters * expansion_f, kernel_size=(1, 1), strides=1, padding='same', name='decode_residual_conv_4')(x)
    x = layers.BatchNormalization(name='decode_residual_bn_4')(x)
    y = layers.add([x, y], name='decode_residual_add_1')
    x = layers.LeakyReLU(name='decode_residual_relu_4')(y)
    y = layers.Conv2D(filters=n_filters, kernel_size=(1, 1), strides=1, padding='same', name='decode_residual_conv_5')(x)
    y = layers.BatchNormalization(name='decode_residual_bn_5')(y)
    y = layers.LeakyReLU(name='decode_residual_relu_5')(y)
    y = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), strides=1,  padding='same', name='decode_residual_conv_6')(y)
    y = layers.BatchNormalization(name='decode_residual_bn_6')(y)
    y = layers.LeakyReLU(name='decode_residual_relu_6')(y)
    y = layers.Conv2D(filters=n_filters * expansion_f, kernel_size=(1, 1), strides=1, padding='same', name='decode_residual_conv_7')(y)
    y = layers.BatchNormalization(name='decode_residual_bn_7')(y)
    y = layers.add([x, y], name='decode_residual_add_2')
    x = layers.LeakyReLU(name='decode_residual_relu_7')(y)
    return x


def _get_upsampled_signal(x, n_filters_lst=[32, 64]):
    x = layers.Conv2DTranspose(filters=n_filters_lst[0], kernel_size=(5, 5), strides=2, padding='same', name='decode_convtrans_1')(x)
    # x = layers.BatchNormalization(name='decode_bn_1')(x)
    x = layers.LeakyReLU(name='decode_relu_1')(x)
    x = layers.Conv2DTranspose(filters=n_filters_lst[1], kernel_size=(3, 3), strides=2, padding='same', name='decode_convtrans_2')(x)
    # x = layers.BatchNormalization(name='decode_bn_2')(x)
    x = layers.LeakyReLU(name='decode_relu_2')(x)
    return x


def _get_encoded_predictions(input_signal):
    x = layers.Dropout(0.25, name='encoded_predictions_dropout_1')(input_signal)
    x = layers.Dense(units=128, activation='relu', name='encoded_predictions_dense')(x)
    x = layers.Dropout(0.25, name='encoded_predictions_dropout_2')(input_signal)
    return x


def get_model():
    img_tensor = layers.Input(shape=(64, 64, 3))
    content_code = _get_content_code(img_tensor)
    style_code = _get_style_code(img_tensor)
    adain_parameters = layers.Lambda(_get_adain_parameters, name='adain_parameters_layer')(style_code)
    x = layers.Lambda(adain, arguments={'epsilon': 1e-5}, name='adain_layer')((content_code, adain_parameters))
    x = _get_decoding_residual_blocks_signal(x, n_filters=16, expansion_f=2)
    x = _get_upsampled_signal(x, n_filters_lst=[16, 32])
    reconstruction = layers.Conv2D(filters=3, kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='reconstruction')(x)
    prediction_code = _get_encoded_predictions(style_code)
    predictions = layers.Dense(units=num_classes, activation='softmax', name='predictions')(prediction_code)
    model = Model(inputs=img_tensor, outputs=[reconstruction, predictions])

    return model
