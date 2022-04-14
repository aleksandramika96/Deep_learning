# Data augmentation is a technique for generating more elements \\
# of training dataset by augmenting samples through random image transformations that \\
# returning images that look plausible. The goal of this solution is that the train model will \\
# never see the same image twice. This attitude allows the model to see more aspects of \\
# processed data and create better generalizations.

# libraries
from keras.preprocessing.image import ImageDataGenerator as IDG

#configuration IDG class
datagen = IDG(
    rotation_range=40, # range of image rotation angles 0-180°
    width_shift_range=0.2, # fraction of total width
    height_shift_range=0.2, # fraction of total height
    shear_range=0.2, # range of random image cropping
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest' # fill newly pixels that may occur from image rotation or move
)
