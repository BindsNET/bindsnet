import os
import sys
import numpy as np

from sklearn.linear_model import LogisticRegression

from bindsnet.datasets import MNIST

# Load MNIST data.
X_train, y_train = MNIST(path=os.path.join('..', 'data')).get_train()
X_test, y_test = MNIST(path=os.path.join('..', 'data')).get_test()

X_train, X_test = X_train.view(60000, -1), X_test.view(10000, -1)

n_samples = 10000

X_train, y_train = X_train[:n_samples], y_train[:n_samples]
X_test, y_test = X_test[:n_samples], y_test[:n_samples]

# Specify logistic regression model
# and fit it to training data.
model = LogisticRegression(verbose=1).fit(X_train, y_train)

# Get predictions on the training data.
predictions = model.predict(X_train)
train_accuracy = 100 * np.mean(y_train.numpy() == predictions)

print('Training accuracy: %.2f' % train_accuracy)

# Get predictions on the test data.
predictions = model.predict(X_test)
test_accuracy = 100 * np.mean(y_test.numpy() == predictions)

print('Test accuracy: %.2f' % test_accuracy)