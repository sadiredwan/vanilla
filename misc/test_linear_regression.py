import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], y)
# plt.show()

from linear_regression import LinearRegression
clf = LinearRegression(learning_rate = 0.1)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

def mse(y_test, y_predicted):
	return np.mean((y_test - y_predicted)**2)

mse_error = mse(y_test, predicted)
print(mse_error)