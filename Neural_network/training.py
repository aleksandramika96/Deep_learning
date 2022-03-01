from keras import optimizers


# optimizer and loss function
def model_optimizer(model):

    return model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss='mse',
                  metrics=['accuracy'])

def fit_model(model, input_tensor, target_tensor):

    return model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)