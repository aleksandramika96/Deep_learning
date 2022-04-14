# Jpg data preprocessing:
# 1. Load data (image files)
# 2. Decode jpg format to pixel rgb format
# 3. Write data into floating point number tensors
# 4. Scale the pixel values from the range 0-255 to 0-1 (neural networks work better with small input values)

# Keras library has tools to perform the conversion process automatically: keras.preprocessing.image, class: ImageDataGenerator

# libraries
from keras.preprocessing.image import ImageDataGenerator as IDG
import configparser
import h5py
from cnn_model_cats_and_dogs import model

config = configparser.ConfigParser()
config.read("configfile.ini")

train_dir = config.get('DATASET_DIR', 'train_dir')
validation_dir = config.get('DATASET_DIR', 'validation_dir')

# scale all data/images by a factor 1/255
train_datagen = IDG(rescale=1. / 255)
test_datagen = IDG(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,  # destination directory
    target_size=(150, 150),  # change the resolution of images to 150x150
    batch_size=20,
    class_mode='binary'  # use function binary_crossentropy as a loss function, we need binary labels
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,  # destination directory
    target_size=(150, 150),  # change the resolution of images to 150x150
    batch_size=20,
    class_mode='binary'  # use function binary_crossentropy as a loss function, we need binary labels
)

for data_batch, labels_batch in train_generator:
    print('kształt danych wsadowych: ', data_batch.shape) # (20, 150, 150, 3)
    print('kształt etykiet danych wsadowych:', labels_batch.shape) # (20,)
    break


# Model fitting using data batch generator
# fit generator - is a method that expects a generator to be defined that returns batches of//
# input data and their labels in an infinity loop
# step per epoch - data are generated infinitely so the keras model has to know how many //
# samples to take from the generator before the end of the epoch
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100, # in our case batches consist of 20 samples, so we need to generate 100 //
                         # batches in order to train the model for 2000 samples
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50) # number of batches to be extracted from the validation data generator

# It is good practice to save all trained models
model.save('trained_models/cats_and_dogs_small_1.h5')