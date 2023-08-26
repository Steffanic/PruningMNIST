import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model


def load_mnist_train_test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def normalize_and_flatten(x_train, x_test):
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train_flat = x_train.reshape((len(x_train), -1))
    x_test_flat = x_test.reshape((len(x_test), -1))
    return x_train_flat, x_test_flat

def build_model(n_hidden_units:list):
    print(f'Building model with {n_hidden_units} hidden units')
    model = Sequential()
    model.add(Dense(n_hidden_units[0], activation='relu', input_shape=(784,)))
    for n_units in n_hidden_units[1:]:
        model.add(Dense(n_units, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def train_model(model, x_train, y_train, x_test, y_test, n_epochs:int, batch_size:int):
    model.compile(optimizer=SGD(lr=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=n_epochs,
              batch_size=batch_size)
    return model, history

def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    return test_loss, test_acc

def build_activation_model(model):
    layer_outputs = [layer.output for layer in model.layers[:-1]]
    activation_model = Model(inputs=model.inputs, outputs=layer_outputs)
    return activation_model

def get_pruned_layer_sizes(activation_model, x_test):
    activations = activation_model(x_test)
    activation_averages = [np.mean(activation, axis=0) for activation in activations]
    activation_masks = [activation_average > 0.5 for activation_average in activation_averages]
    pruned_layer_sizes = [np.sum(mask) for mask in activation_masks]
    return pruned_layer_sizes

def plot_histories(histories):
    for i, history in enumerate(histories):
        plt.plot(history.history['val_accuracy'], label=f'val_acc - {i}')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

def main():
    (x_train, y_train), (x_test, y_test) = load_mnist_train_test()
    x_train_flat, x_test_flat = normalize_and_flatten(x_train, x_test)

    histories = []
    hidden_layer_sizes = [64, 64, 64, 64]
    for i in range(10):
        model = build_model(hidden_layer_sizes)
        model, history = train_model(model, x_train_flat, y_train, x_test_flat, y_test, 10, 256)
        test_loss, test_acc = evaluate_model(model, x_test_flat, y_test)
        print('Test accuracy:', test_acc)
        histories.append(history)
        activation_model = build_activation_model(model)
        pruned_layer_sizes = get_pruned_layer_sizes(activation_model, x_test_flat)
        hidden_layer_sizes = pruned_layer_sizes
    plot_histories(histories)

if __name__ == '__main__':
    main()
