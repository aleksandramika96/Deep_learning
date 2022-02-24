# -*- coding: utf-8 -*-

# packages
import configparser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers

config = configparser.ConfigParser()
config.read("config.ini")

sections = config.sections()
# config.options(sections[0])
# config.get(sections[0], config.options(sections[0]))

class SequentialModel():
    """ A sequential model is appropriate for a plain stack of layers where each layer has excatly one input tensor and one output tensor """

    # Define sequential model
    # for i in range(1,config.get('Sequential Model', layers)+1):
    #     exec(f'layer_{i} = {i}')
    model = keras.Seuential(
        [
            layers.Dense(2, activation="relu", name="layer1") # relu - nonlinear function
            layers.Dense(3, activation="relu", name="layer2")
            layers.Dense(4, name="layer3")

        ]
    )

    # Call model on a test input
    x = tf.ones((3,3))
    y=model(x)
    # print('Number of weights after calling the model:', len(model.weights))

class FunctionalAPI():