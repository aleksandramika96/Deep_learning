from keras.preprocessing.image import ImageDataGenerator as IDG
from cnn_model_dropout_cats_and_dogs import model
import configparser
config = configparser.ConfigParser()
config.read("configfile.ini")
train_dir = config.get('DATASET_DIR', 'train_dir')
validation_dir = config.get('DATASET_DIR', 'validation_dir')

train_datagen = IDG(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# remember to not modify the validation data!
test_datagen = IDG(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('trained_models/cats_and_dogs_small_2.h5')