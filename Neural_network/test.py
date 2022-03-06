# d= dict()
# for i in range(1,5+1):
#     d['string{}'.format(i)]=i
# print(d)
#
#
# for k in range(5):
#     exec(f'cat_{k} = k*2')
# print(cat_1)
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import models, layers
# model = models.Sequential()
# model.add(layers.Dense(32, activation="relu", name="layer1", input_shape=(784,)))
# model.add(layers.Dense(10, activation="softmax"))
# model = keras.Sequential(
#     [
#         layers.Dense(2, activation="relu", name="layer1"), # relu - nonlinear function
#         layers.Dense(3, activation="relu", name="layer2"),
#         layers.Dense(4, name="layer3")
#
#     ]
# )
# x = tf.ones((3, 3))
# y = model(x)


import configparser
config = configparser.ConfigParser()
config.read("config.ini")

sections = config.sections()
print(config.get('Input and output tensors', 'input_tensor_shape'))
# config.get(sections[0], config.options(sections[0]))