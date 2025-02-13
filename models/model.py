import tensorflow as tf
from Encoder import Encoder
from tensorflow import keras
import numpy as np
from sklearn.metrics import mean_squared_error
from Mutil_scale_prot import MultiScaleConvA
from tensorflow.keras.regularizers import l2


def pseudo_huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    delta_squared = delta ** 2
    huber_loss = delta_squared * (tf.sqrt(1 + (error / delta) ** 2) - 1)
    return huber_loss


def get_model():
    inputESM = tf.keras.layers.Input(shape=(6, 1280))
    inputProt = tf.keras.layers.Input(shape=(161, 1024))

    sequence = tf.keras.layers.Dense(512)(inputESM)
    sequence = tf.keras.layers.Dense(256)(sequence)
    sequence = tf.keras.layers.Dense(128)(sequence)
    sequence = Encoder(1, 128, 8, 1024, rate=0.3)(sequence)
    sequence = tf.keras.layers.Flatten()(sequence)

    sequence_prot = tf.keras.layers.Dense(512)(inputProt)
    sequence_prot = tf.keras.layers.Dense(256)(sequence_prot)
    sequence_prot = tf.keras.layers.Dense(128)(sequence_prot)
    Prot = MultiScaleConvA()(sequence_prot)

    sequenceconcat = tf.keras.layers.Concatenate(name='concatenate')([sequence, Prot])

    l2_reg = 1e-3  
    feature = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(sequenceconcat)
    feature = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(feature)
    feature = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(feature)
    feature = tf.keras.layers.Dropout(0.4)(feature)

    y = tf.keras.layers.Dense(1)(feature)
    qa_model = tf.keras.models.Model(inputs=[inputESM, inputProt],
                                     outputs=y)  
    adam = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0,
                                    clipvalue=0.5)  

    delta = 1
    qa_model.compile(optimizer=adam, loss=lambda y_true, y_pred: pseudo_huber_loss(y_true, y_pred, delta=delta),
                     metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError()])

    qa_model.summary()  
    return qa_model 
