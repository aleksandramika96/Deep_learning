# libraries
import os, shutil

# copy images to sets: train, validation, test
original_dataset_dir = '/dataset'
base_dir = '/dataset_cats_and_dogs_small' # target directory where smaller data sets will be placed
os.mkdir(base_dir)
