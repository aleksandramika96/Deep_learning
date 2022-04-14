import os

import matplotlib.pyplot as plt
from keras.preprocessing import image
from augmentation import datagen
import configparser
config = configparser.ConfigParser()
config.read("configfile.ini")

train_cats_dir = config.get('DATASET_DIR', 'train_cats_dir')
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

img_path = fnames[3]  # selecting image to modify
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img) # numpy array with 3 dimensions (150,150,3)
x = x.reshape((1,) + x.shape) # change the size to (1, 150, 150, 3)
i = 0

# flow loop generate randomly modified image batches
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
# 3 transformations

plt.show()
