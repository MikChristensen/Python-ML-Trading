import random
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.optimizers import Adam


class get_MLP_Models():
    
    def __init__(self):
        self.set_seeds(100)
    
    def set_seeds(self, seed = 100):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def create_model_1(self, cols):
        optimizer = Adam(learning_rate=0.001)
        model = Sequential([
        Dense(len(cols) * 6, activation="relu", input_shape=(len(cols),)),

        Dense(len(cols) * 9, activation="relu"),

        Dense(len(cols) * 12, activation="relu"),

        Dense(len(cols), activation="relu"),

        Dense(int(len(cols) / 2), activation="relu"),

        Dense(int(len(cols) / 3), activation="relu"),

        Dense(int(len(cols) / 4), activation="relu"),

        Dense(int(len(cols) / 5), activation="relu"),

        Dense(int(len(cols) / 6), activation="relu"),

        Dense(len(cols), activation="relu"),
        Dense(1)
        ])

        model.add(Dense(1, use_bias=True, activation = "sigmoid"))

        print(model.summary())

        model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['acc', 'mse'])
        return model
    
    
    
    def create_model_1():
        optimizer = Adam(learning_rate=0.001)
        model = Sequential([
        Dense(len(cols) * len(cols), activation="relu", input_shape=(len(cols),)),

        Dense(len(cols) * 9, activation="relu"),

        Dense(len(cols) * 12, activation="relu"),

        Dense(len(cols), activation="relu"),


        Dense(len(cols), activation="relu"),
        Dense(1)
        ])

        model.add(Dense(1, use_bias=True, activation = "sigmoid"))

        print(model.summary())

        model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['acc', 'mse'])
        return model

