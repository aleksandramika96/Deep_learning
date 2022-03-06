from tensorflow.keras import optimizers
from keras.datasets import imdb
import numpy as np
import neural_model as nm
import matplotlib.pyplot as plt


# optimizer and loss function
def model_optimizer(model):
    """
    args:
        optimizer: rmsprop,
        loss function: binary_crossentropy
    """
    return model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
                         loss='binary_crossentropy',  # mse
                         metrics=['accuracy'])


def fit_model(model, input_tensor, target_tensor, x_val, y_val):
    return model.fit(input_tensor, target_tensor, batch_size=512, epochs=20, validation_data=(x_val, y_val))


def load_imbd_dataset(num_words=10000):
    """
    loading imbd dataset

    args:
        num_words - number of retained words in the set, occurring the most frequently

    outputs:
        train_data, test_data - list of reviews
        train_labels, test_labels - list of index reviews, labels of zeroes (negative review) and ones (positive review)
    """
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)
    return (train_data, train_labels), (test_data, test_labels)


def vectorize_sequences(sequences, dimension=10000):
    """ One-Hot Vectors Encoding """

    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def plot_training_and_validation_loss_graph(history):
    """ Visualization the training loss vs validation loss"""
    history_dict = history.history
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    accuracy = history_dict['accuracy']
    # val_accuracy = history_dict['val_accuracy']

    epochs = range(1, len(accuracy)+1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def plot_training_and_validation_accuracy_graph(history):
    """ Visualization the training accuracy vs validation accuracy"""
    history_dict = history.history
    # loss = history_dict['loss']
    # val_loss = history_dict['val_loss']
    accuracy = history_dict['accuracy']
    val_accuracy = history_dict['val_accuracy']

    epochs = range(1, len(accuracy) + 1)

    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


(train_data, train_labels), (test_data, test_labels) = load_imbd_dataset()

# samples
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# converting sample labels into vectors
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

sm_model = nm.SequentialModel()
model = sm_model.model

# configuring model, optimizer: rmsprop, loss function: binary_crossentropy
model_optimizer(model)

# model validation, creating validation dataset
x_val = x_train[:sm_model.input_tensor_shape]
partial_x_train = x_train[sm_model.input_tensor_shape:]

y_val = y_train[:sm_model.input_tensor_shape]
partial_y_train = y_train[sm_model.input_tensor_shape:]

history = fit_model(model, input_tensor=partial_x_train, target_tensor=partial_y_train, x_val=x_val, y_val=y_val)
# history_dict = history.history
# print(history_dict.keys())  # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

plot_training_and_validation_accuracy_graph(history)
plot_training_and_validation_loss_graph(history)