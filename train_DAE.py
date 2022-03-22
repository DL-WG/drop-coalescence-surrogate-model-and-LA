import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image
import glob
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from joblib import Parallel, delayed

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Convolution2D, MaxPooling2D, UpSampling2D,Cropping2D, LSTM, RepeatVector,Activation
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def get_compiled_model():
    #CAE
    input_img = Input(shape=(1024, 1024, 1))
    x = Convolution2D(8, (16, 16), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((8, 8), padding='same')(x)
    x = Convolution2D(16, (8, 8), activation='relu', padding='same')(x)
    x = MaxPooling2D((8, 8), padding='same')(x)
    x = Convolution2D(32, (4, 4), activation='relu', padding='same')(x)
    x = MaxPooling2D((4, 4), padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(16)(x)

    encoder = Model(input_img, encoded)
    encoder.summary()

    decoder_input = Input(shape=(16))
    x = Dense(512)(decoder_input)
    x = Reshape((4,4,32))(x)
    x = Convolution2D(32, (4, 4), activation='relu', padding='same')(x)
    x = UpSampling2D((4, 4))(x)
    x = Convolution2D(16, (8, 8), activation='relu', padding='same')(x)
    x = UpSampling2D((8, 8))(x)
    x = Convolution2D(8, (16, 16), activation='relu', padding='same')(x)
    x = UpSampling2D((8, 8))(x)
    decoded = Convolution2D(1, (16, 16), activation='sigmoid', padding='same')(x)

    decoder = Model(decoder_input, decoded)
    decoder.summary()

    encoder=Model(inputs=input_img, outputs=encoded, name = 'encoder')
    decoder=Model(inputs=decoder_input, outputs=decoded, name = 'decoder')
    autoencoder_outputs = decoder(encoder(input_img))
    autoencoder= Model(input_img, autoencoder_outputs, name='autoencoder')
    autoencoder.summary()
    #Optimizers 
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    autoencoder.compile(optimizer=opt, loss="binary_crossentropy")
    return autoencoder

def get_dataset():
    batch_size = 150

    pic_array_ens = np.load('pic_array_ens.npy') # The collection of all images is called pic_array_ens
    num_array = pic_array_ens.shape[0]
    select_index = list(set(range(0,num_array,1))-set(range(0,num_array,3)))
    print('shape before: ', pic_array_ens.shape[0])
    pic_array_ens = pic_array_ens[select_index]
    num_array = pic_array_ens.shape[0]
    train_index = list(set(range(0,num_array,1))- set(range(0,num_array,5)))
    np.random.shuffle(train_index)
    test_index = list(set(range(0,num_array,5)))
    x_train = pic_array_ens[train_index]
    x_test = pic_array_ens[test_index]
    print('Train shape after: %d, Test shape after: %d' % (x_train.shape[0],x_test.shape[0]))
    print('Shape of train & test before reshape:',x_train.shape,x_test.shape)
    x_train = np.reshape(x_train,(len(train_index),1024,1024,1))
    x_test = np.reshape(x_test, (len(test_index),1024,1024,1))
    x_train, x_test = x_train.astype('float16'), x_test.astype('float16')
    print('Shape of train & test after  reshape:',x_train.shape,x_test.shape)
    return (
        tf.data.Dataset.from_tensor_slices((x_train, x_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(batch_size),
    )

def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    print("checkpoints: ", checkpoints)
    print("local dir: ", os.getcwd())
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()


def run_training(epochs=1):
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()

    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        autoencoder = make_or_restore_model()

    callbacks = [
        # This callback saves a SavedModel every epoch
        # We include the current epoch in the folder name.
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/checkpoint-{epoch}", save_freq="epoch"
        )
    ]
    train_dataset, val_dataset = get_dataset()
    autoencoder.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
        verbose=2,
    )

checkpoint_dir = './checkpoint'
# Running the first time creates the model
run_training(epochs=1500)
