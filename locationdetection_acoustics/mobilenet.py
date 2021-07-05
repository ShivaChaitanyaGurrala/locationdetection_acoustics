import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def pretrained_model():
    img_size = (224, 224)
    img_shape = img_size + (3,)
    initializer = tf.keras.initializers.he_normal(seed=32)
    '''base_model = tf.keras.applications.vgg19.VGG19(input_shape=img_shape,
                                                   include_top=False,
                                                   weights='imagenet')'''
    '''base_model = tf.keras.applications.DenseNet169(input_shape=img_shape,
                                                   include_top=False,
                                                   weights='imagenet')'''
    base_model = tf.keras.applications.DenseNet201(input_shape=img_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    '''base_model = tf.keras.applications.resnet.ResNet50(input_shape=img_shape,
                                                       include_top=False,
                                                       weights='imagenet')'''
    base_model.trainable = False
    for layer in base_model.layers:
        if 'conv5' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    x = base_model.output
    x = layers.Flatten()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(units=256,
                     activation='relu',
                     # kernel_initializer=initializer
                     )(x)

    x = layers.Dropout(0.4)(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(units=128,
                     activation='relu',
                     # kernel_initializer=initializer
                     )(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(units=10,
                     activation='softmax',
                     # kernel_initializer=initializer
                     )(x)

    # use for VGG
    # x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Dense(4096, activation='relu')(x)
    # x = layers.Dense(10, activation='softmax')(x)

    pretrained = models.Model(inputs=base_model.input, outputs=x)
    pretrained.summary()
    return pretrained
