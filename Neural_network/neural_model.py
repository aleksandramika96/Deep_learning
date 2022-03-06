# -*- coding: utf-8 -*-

# packages
import configparser
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers

config = configparser.ConfigParser()
config.read("config.ini")

sections = config.sections()
# config.options(sections[0])
# config.get(sections[0], config.options(sections[0]))

class SequentialModel:
    """
        A sequential model is appropriate for a plain stack of layers where each layer
        has exactly one input tensor and one output tensor
    """
    input_tensor_shape=int(config.get('Input and output tensors', 'input_tensor_shape'))
    # Define sequential model
    # for i in range(1,config.get('Sequential Model', layers)+1):
    #     exec(f'layer_{i} = {i}')
    model = keras.Sequential(
        [
            layers.Dense(16, activation="relu", name="layer1", input_shape=(input_tensor_shape,)), # relu - nonlinear function
            layers.Dense(16, activation="relu", name="layer2"),
            layers.Dense(1, activation="sigmoid", name="layer3")
        ]
    )
    number_of_weights = len(model.weights)
    model_summary = model.summary

    # Call model on a test input
    # x = tf.ones((3,3))
    # y=model(x)
    # print('Number of weights after calling the model:', len(model.weights))


class FunctionalAPI:
    """ A functional API is a way to create models that are more flexible than the sequential API. It can handle models with non-linear topology, shared layers, and multiple inputs and outputs.
        The main idea is that a DL model is usually a directed acyclic graph (DAG) of layers. The functional API is a way to build graphs of layers.
     """
    input_tensor_shape = int(config.get('Input and output tensors', 'input_tensor_shape'))
    input_tensor = layers.Input(shape=(input_tensor_shape,))
    y = layers.Dense(16, activation="relu")(input_tensor)
    x = layers.Dense(16, activation="relu")(y)
    output_tensor = layers.Dense(1, activation="softmax")(x)

    model = keras.Model(inputs=input_tensor, outputs=output_tensor)
    model_summary = model.summary