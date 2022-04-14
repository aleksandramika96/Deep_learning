# Generate plot of changes in loss and validation values while training model

# libraries
import matplotlib.pyplot as plt
from cnn_load_and_train_data_cats_and_dogs import history

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy', color='orange')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy', color='lightgrey')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('training_validation_acc.png', dpi=100)
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss', color='orange')
plt.plot(epochs, val_loss, 'b', label='Validation loss', color='lightgrey')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('training_validation_loss.png', dpi=100)
