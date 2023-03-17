from __future__ import print_function

import os, sys, pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from loss_function import *
from train_generator import *
from val_generator import *
from fcn import *


tf.config.list_physical_devices('GPU')

metric = SemanticLossFunction()
model_loaded = FCNModel()
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def train():

    train_images_number = 5
    val_images_number = 2
    train_dataset = TrainGenerator(train_images_number)
    val_dataset = ValGenerator(val_images_number)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    """ Hyperparameters """
    input_shape = (512, 512, 4)
    lr = 1e-4
    _epochs = 50
    _loss=metric.jacard_coef_loss
    _metrics=[metric.jacard_coef, metric.dice_coef, metric.sensitivity, metric.specificity]

    model = model_loaded.build_fcn(input_shape)
    model.compile(optimizer=Adam(learning_rate=lr), loss=_loss, metrics=_metrics)

    # Callbacks
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    history = model.fit(train_dataset, epochs=_epochs, verbose=1,
                        validation_data=val_dataset,
                        callbacks=[model_checkpoint])

    return history

if __name__ == '__main__':
    history = train()
    