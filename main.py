import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras

import config
from misc import PSNR, image_generator, get_images_list


def _conv2d(
    x,
    filters=None,
    kernel_size=None, 
    strides=(1, 1), 
    padding='same', 
    kernel_initializer='he_normal', 
    activation='relu',
    name=None):

    x = keras.layers.Conv2D(filters, kernel_size, strides, padding, kernel_initializer=kernel_initializer, name=name)(x)
    x = keras.layers.Activation(activation=activation, name=name + '_activation')(x)

    return x


def _get_model(mode='train'):
    input_image = keras.Input(shape=(41, 41, 1))
    
    x = _conv2d(input_image, 64, (3, 3), name='conv2d_0')
    
    for i in range(1, 18):
        x = _conv2d(x, 64, (3, 3), name='conv2d_{}'.format(i))

    x = keras.layers.Conv2d(1, (3, 3), padding='same', kernel_initializer='he_normal')(x)

    output_image = keras.layers.add([x, input_image]) if mode == 'train' else keras.layers.merge([x, input_image])
    model = keras.models.Model(input_image, output_image)

    return model


def train():
    model = _get_model()
    optimizer = keras.optimizers.Adam(lr=0.0001)

    model.compile(optimizer, loss='mse', metrics=[PSNR, 'accuracy'])
    model.summary()

    train_list = get_images_list(config.data_path, scales=config.TRAIN_SCALES)
    val_list = get_images_list(config.validation_path, scales=config.VAL_SCALES)

    filepath = './checkpoint/weights-{epochs:02d}-{PSNR:.2f}.hdf5'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor=PSNR, verbose=1, mode='max')

    callbacks_list = [checkpoint]
    
    model.fit_generator(
        image_generator(train_list), 
        steps_per_epoch=len(train_list) // config.BATCH_SIZE, 
        validation_data=image_generator(val_list),
        validation_steps=len(train_list) // config.BATCH_SIZE,
        epochs=config.EPOCHS,
        workers=8,
        callbacks=callbacks_list)

    print('Training Complete. \nSaving final model.')
    
    model.save('model.h5')

def test(path):
    model = _get_model(mode='test')
    model.load_weights(config.model_path)

    image = keras.preprocessing.image.load_img(path, grayscale=True, target_size=(41, 41, 1))
    
    x = keras.preprocessing.image.img_to_array(image)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    test_image = np.reshape(pred, (41, 41))

    cv2.imshow("image", test_image)