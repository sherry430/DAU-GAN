from __future__ import division
import os
import os.path
from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, concatenate, Reshape, Dropout, Lambda, Permute, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Activation, BatchNormalization, Conv2DTranspose, Add, multiply

os.environ['KERAS_BACKEND'] = 'tensorflow'


def attach_attention_module(dual_attention_feature, ratio=8):
    """
    dual-attention module
    """
    dual_attention_feature = channel_attention(dual_attention_feature, ratio)
    dual_attention_feature = spatial_attention(dual_attention_feature)
    return dual_attention_feature


def channel_attention(input_feature, ratio=8):
    """
    channel attention module
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    dual_attention_feature = Add()([avg_pool, max_pool])
    dual_attention_feature = Activation('sigmoid')(dual_attention_feature)

    if K.image_data_format() == "channels_first":
        dual_attention_feature = Permute((3, 1, 2))(dual_attention_feature)

    return multiply([input_feature, dual_attention_feature])


def spatial_attention(input_feature, kernel_size=7):
    """
    spatial attention module
    """
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        dual_attention_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        dual_attention_feature = input_feature
    avg_pool = Lambda(lambda x: K.mean(
        x, axis=3, keepdims=True))(dual_attention_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(
        x, axis=3, keepdims=True))(dual_attention_feature)
    assert max_pool._keras_shape[-1] == 1
    # concat = Concatenate(axis=3)([avg_pool, max_pool])
    concat = concatenate([avg_pool, max_pool], axis=3)
    assert concat._keras_shape[-1] == 2
    dual_attention_feature = Conv2D(filters=1,
                                    kernel_size=kernel_size,
                                    strides=1,
                                    padding='same',
                                    activation='sigmoid',
                                    kernel_initializer='he_normal',
                                    use_bias=False)(concat)
    assert dual_attention_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        dual_attention_feature = Permute((3, 1, 2))(dual_attention_feature)

    return multiply([input_feature, dual_attention_feature])


def conv_block_down(inputs, feature_channel, kernel_size, strides, padding, drop_rate=0):
    conv = Conv2D(feature_channel, kernel_size=kernel_size, strides=strides, padding=padding,
                  kernel_initializer='he_normal')(inputs)
    conv = BatchNormalization(scale=False, axis=3)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(feature_channel, kernel_size=kernel_size, padding=padding,
                  kernel_initializer='he_normal')(conv)
    conv = BatchNormalization(scale=False, axis=3)(conv)
    conv = Activation('relu')(conv)
    drop = Dropout(drop_rate)(conv)
    pool = MaxPooling2D(pool_size=(2, 2))(drop)
    return drop, pool


def conv_block_up(inputs, skip_connection_inputs, feature_channel, kernel_size, padding):
    up = Conv2DTranspose(feature_channel, kernel_size=2, strides=2,
                         padding=padding, kernel_initializer='he_normal')(inputs)
    up = BatchNormalization(axis=3)(up)
    up = Activation('relu')(up)
    x1 = attach_attention_module(skip_connection_inputs)
    merge = concatenate([x1, up], axis=3)
    conv = Conv2D(feature_channel, kernel_size, padding=padding,
                  kernel_initializer='he_normal')(merge)
    conv = BatchNormalization(scale=False, axis=3)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(feature_channel, kernel_size, padding=padding,
                  kernel_initializer='he_normal')(conv)
    conv = BatchNormalization(scale=False, axis=3)(conv)
    conv = Activation('relu')(conv)
    return conv


def generator_da_unet(img_size, n_filters=64, name='g'):
    """
    generate network based on unet with dual attention module
    """
    # set image specifics
    k = 3  # kernel size
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size, img_size
    padding = 'same'
    inputs = Input((img_height, img_width, img_ch))

    conv1, pool1 = conv_block_down(inputs, feature_channel=n_filters, kernel_size=(
        k, k), strides=(1, 1), padding=padding, drop_rate=0)
    conv2, pool2 = conv_block_down(pool1, feature_channel=2*n_filters,
                                   kernel_size=(k, k), strides=(1, 1), padding=padding, drop_rate=0)
    conv3, pool3 = conv_block_down(pool2, feature_channel=4*n_filters,
                                   kernel_size=(k, k), strides=(1, 1), padding=padding, drop_rate=0.5)

    conv4_1, _ = conv_block_down(pool3, feature_channel=8*n_filters,
                                 kernel_size=(k, k), strides=(1, 1), padding=padding, drop_rate=0.5)
    conv4_2, _ = conv_block_down(conv4_1, feature_channel=8*n_filters,
                                 kernel_size=(k, k), strides=(1, 1), padding=padding, drop_rate=0.5)
    merge_dense = concatenate([conv4_2, conv4_1], axis=3)
    conv4_3, _ = conv_block_down(merge_dense, feature_channel=8*n_filters,
                                 kernel_size=(k, k), strides=(1, 1), padding=padding, drop_rate=0.5)

    conv5 = conv_block_up(conv4_3, conv3, feature_channel=4 *
                          n_filters, kernel_size=(k, k), padding=padding)
    conv6 = conv_block_up(conv5, conv2, feature_channel=2 *
                          n_filters, kernel_size=(k, k), padding=padding)
    conv7 = conv_block_up(
        conv6, conv1, feature_channel=n_filters, kernel_size=(k, k), padding=padding)

    conv7 = Conv2D(2, (k, k), activation='relu', padding=padding,
                   kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)

    outputs = Conv2D(out_ch, (1, 1), padding=padding,
                     activation='sigmoid')(conv7)
    print('output_g:', outputs.shape)
    g = Model(inputs, outputs, name=name)
    return g


def discriminator(img_size, n_filters, init_lr, name='d'):
    """
    discriminator network
      stride 2 conv X 2
        max pooling X 4
    fully connected X 1
    """

    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size, img_size
    padding = 'same'  # 'valid'

    inputs = Input((img_height, img_width, img_ch + out_ch))
    _, pool1 = conv_block_down(inputs, n_filters, kernel_size=(
        k, k), strides=(s, s), padding=padding, drop_rate=0)
    _, pool2 = conv_block_down(
        pool1, 2*n_filters, kernel_size=(k, k), strides=(s, s), padding=padding, drop_rate=0)
    _, pool3 = conv_block_down(
        pool2, 4*n_filters, kernel_size=(k, k), strides=(1, 1), padding=padding, drop_rate=0)
    _, pool4 = conv_block_down(
        pool3, 8*n_filters, kernel_size=(k, k), strides=(1, 1), padding=padding, drop_rate=0)
    _, pool5 = conv_block_down(
        pool4, 16*n_filters, kernel_size=(k, k), strides=(1, 1), padding=padding, drop_rate=0)

    gap = GlobalAveragePooling2D()(pool5)
    outputs = Dense(1, activation='sigmoid')(gap)
    print('output_d:', outputs.shape)
    d = Model(inputs, outputs, name=name)
    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5),
              loss='binary_crossentropy', metrics=['accuracy'])
    return d


def GAN(g, d, img_size, alpha_recip, init_lr, name='gan'):
    """
    GAN (that binds generator and discriminator)
    """
    img_ch = 3
    seg_ch = 1
    img_height, img_width = img_size, img_size

    raw_liver = Input((img_height, img_width, img_ch))
    real_vessel = Input((img_height, img_width, seg_ch))
    fake_vessel = g(raw_liver)
    fake_pair = concatenate([raw_liver, fake_vessel], axis=3)

    gan = Model([raw_liver, real_vessel], d(fake_pair), name=name)

    def gan_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)
        L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)

        real_vessel_flat = K.batch_flatten(real_vessel)
        fake_vessel_flat = K.batch_flatten(fake_vessel)
        L_seg = objectives.binary_crossentropy(
            real_vessel_flat, fake_vessel_flat)
        return alpha_recip*L_adv + L_seg

    gan.compile(optimizer=Adam(lr=init_lr, beta_1=0.5),
                loss=gan_loss, metrics=['accuracy'])

    return gan
