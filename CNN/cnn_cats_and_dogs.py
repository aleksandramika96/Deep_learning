# libraries
import os, shutil
import configparser

config = configparser.ConfigParser()
config.add_section('DATASET_DIR')

# copy images to sets: train, validation, test
original_dataset_dir = os.path.join(os.getcwd(), 'dataset')
base_dir = os.path.join(os.getcwd(), 'dataset_cats_and_dogs_small')  # target directory where smaller data sets will be placed
os.mkdir(base_dir)
config.set('DATASET_DIR', 'BASE_DIR', base_dir)

# subdirectory of train, validation, and test sets
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
config.set('DATASET_DIR', 'TRAIN_DIR', train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
config.set('DATASET_DIR', 'VALIDATION_DIR', validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
config.set('DATASET_DIR', 'TEST_DIR', test_dir)

# cats
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
config.set('DATASET_DIR', 'TRAIN_CATS_DIR', train_cats_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
config.set('DATASET_DIR', 'VALIDATION_CATS_DIR', validation_cats_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
config.set('DATASET_DIR', 'TEST_CATS_DIR', test_cats_dir)

# dogs
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
config.set('DATASET_DIR', 'TRAIN_DOGS_DIR', train_dogs_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)
config.set('DATASET_DIR', 'VALIDATION_DOGS_DIR', validation_dogs_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)
config.set('DATASET_DIR', 'TEST_DOGS_DIR', test_dogs_dir)

# copy first 3000 cats images to train_cats_dir directory
fnames = ['cat.{}.jpg'.format(i) for i in range(1, 3001)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, 'training_set/cats', fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# copy next 1000 cats images to validation_cats_dir directory
fnames = ['cat.{}.jpg'.format(i) for i in range(3001, 4001)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, 'training_set/cats', fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# copy next 1000 cats images to test_cats_dir directory
fnames = ['cat.{}.jpg'.format(i) for i in range(4001, 5001)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, 'test_set/cats', fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)


# copy first 3000 dogs images to train_dogs_dir directory
fnames = ['dog.{}.jpg'.format(i) for i in range(1, 3001)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, 'training_set/dogs', fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# copy next 1000 dogs images to validation_dogs_dir directory
fnames = ['dog.{}.jpg'.format(i) for i in range(3001, 4001)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, 'training_set/dogs', fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

# copy next 1000 dogs images to test_dogs_dir directory
fnames = ['dog.{}.jpg'.format(i) for i in range(4001, 5001)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, 'test_set/dogs', fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print('liczba obrazów treningowych kotów:', len(os.listdir(train_cats_dir)))
print('liczba obrazów walidacyjnych kotów:', len(os.listdir(validation_cats_dir)))
print('liczba obrazów testowych kotów:', len(os.listdir(test_cats_dir)))

print('liczba obrazów treningowych psow:', len(os.listdir(train_dogs_dir)))
print('liczba obrazów walidacyjnych psow:', len(os.listdir(validation_dogs_dir)))
print('liczba obrazów testowych psow:', len(os.listdir(test_dogs_dir)))


# write the new structure to the configuration file
with open(os.path.join(os.getcwd(), 'configfile.ini'), 'w') as configfile:
    config.write(configfile)