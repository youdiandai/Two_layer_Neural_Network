import copy
import os

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from tqdm import tqdm


# scroll to the bottom to start coding your solution


def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):
    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


class OneLayerNeural:
    def __init__(self, n_features, n_classes):
        self.a_value = None
        self.n_features = n_features
        self.n_classes = n_classes
        # weights_biases = xavier(n_features + 1, n_classes)
        self.weights, self.biases = xavier(n_features, n_classes), xavier(1, n_classes)
        # self.bias = np.zeros(n_classes)
        # Initiate weights and biases using Xavier

    def forward(self, X):
        self.z_value = np.dot(X, self.weights) + self.biases
        self.a_value = sigmoid(np.dot(X, self.weights) + self.biases)
        # Perform a forward step
        return self.a_value

    def backprop(self, X, y, alpha):
        mse_delta = mse_derivatives(self.a_value, y)
        sigma_delta = sigma_derivatives(self.z_value)
        weights_gradinents = X.T @ (sigma_delta * mse_delta)
        weights_gradinent = weights_gradinents / len(X)

        biases_gradinents = sum(mse_delta * sigma_delta) / len(X)
        # Updating  weights and biases.
        self.weights = self.weights - alpha * weights_gradinent
        self.biases = self.biases - alpha * biases_gradinents


def epoch_train(estimator, alpha, X, y, batch_size):
    batchs = []
    new_X = copy.copy(X)
    new_y = copy.copy(y)
    for x in range(int(np.ceil(len(new_X) / batch_size))):
        batchs.append((new_X[:batch_size], new_y[:batch_size]))
        new_X = new_X[batch_size:]
        new_y = new_y[batch_size:]
    for _X, _y in batchs:
        estimator.forward(_X)
        estimator.backprop(_X, _y, alpha)
    estimator.forward(X)
    return mse(estimator.a_value, y)


def mse(y_pred, y_true):
    # print(y_true.size)
    return np.power(y_pred - y_true, 2).sum() / y_true.size


def accuracy(estimator, X, y):
    estimator.forward(X)
    y_pred = [x.argmax() for x in estimator.a_value]
    y_true = [x.argmax() for x in y]
    tp = len([1 for x, y in zip(y_pred, y_true) if x == y])
    return tp / len(y_true)


def sigmoid(x):
    return 1 / (1 + pow(np.e, -x))


def scale(X_train, X_test):
    return X_train / X_train.max(), X_test / X_test.max()


def xavier(n_in, n_out):
    low, high = -(np.power(6, 1 / 2) / np.power(n_in + n_out, 1 / 2)), np.power(6, 1 / 2) / np.power((n_in + n_out),
                                                                                                     1 / 2)
    return np.random.uniform(low, high, (n_in, n_out))  # resulting matrix of weights


def mse_derivatives(y_pred, y_true):
    return 2 * (y_pred - y_true)


def sigma_derivatives(x):
    return sigmoid(x) * (1 - sigmoid(x))


class TwoLayerNeural():
    def __init__(self, n_features, n_classes):
        # Initializing weights
        self.hidden_layer = OneLayerNeural(n_features, 64)
        self.output_layer = OneLayerNeural(64, n_classes)

    def forward(self, X):
        self.hidden_layer.forward(X)
        self.output_layer.forward(self.hidden_layer.a_value)
        self.a_value = self.output_layer.a_value
        return self.output_layer.a_value

    def backprop(self, X, y, alpha):
        # Calculating gradients for each of weights and biases.
        out_mse_delta = mse_derivatives(self.output_layer.a_value, y)
        out_sigma_delta = sigma_derivatives(self.output_layer.z_value)
        out_weights_gradinents = self.hidden_layer.a_value.T @ (out_sigma_delta * out_mse_delta)
        out_weights_gradinent = out_weights_gradinents / X.shape[0]
        out_biases_gradinents = sum(out_mse_delta * out_sigma_delta) / X.shape[0]

        ol_ad = self.output_layer.weights @ ((out_mse_delta * out_sigma_delta).T)
        hidden_sigma_delta = sigma_derivatives(self.hidden_layer.z_value)
        hidden_weights_gradinents = X.T @ (hidden_sigma_delta * ol_ad.T)
        hidden_weights_gradinent = hidden_weights_gradinents / X.shape[0]
        hidden_biases_gradinents = sum(hidden_sigma_delta * ol_ad.T) / X.shape[0]
        # Updating  weights and biases.
        self.output_layer.weights = self.output_layer.weights - alpha * out_weights_gradinent
        self.output_layer.biases = self.output_layer.biases - alpha * out_biases_gradinents
        self.hidden_layer.weights = self.hidden_layer.weights - alpha * hidden_weights_gradinent
        self.hidden_layer.biases = self.hidden_layer.biases - alpha * hidden_biases_gradinents


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    X_train, X_test = scale(X_train, X_test)

    nn = TwoLayerNeural(X_train.shape[1], y_train.shape[1])

    loss_logging = []
    accuracy_logging = []
    for x in tqdm(range(20)):
        loss_logging.append(epoch_train(nn, 0.5, X_train, y_train, 100))
        accuracy_logging.append(accuracy(nn, X_test, y_test))
    plot(loss_logging, accuracy_logging)
    print(accuracy_logging)
