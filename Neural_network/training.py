from keras import optimizers
from keras.datasets import imdb
import numpy as np
import neural_model as nm


# optimizer and loss function
def model_optimizer(model):

    return model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss='mse',
                  metrics=['accuracy'])

def fit_model(model, input_tensor, target_tensor):

    return model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)

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


(train_data, train_labels), (test_data, test_labels) = load_imbd_dataset()

# samples
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# converting sample labels into vectors
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = nm.SequentialModel.model
