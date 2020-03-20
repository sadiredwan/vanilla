import numpy as np

class SVM:

	def __init__(self, learning_rate = 0.001, lambda_param = 0.01, n_iters=1000):
		self.lr = learning_rate
		self.lambda_param = lambda_param
		self.n_iters = n_iters
		self.w = None
		self.b = None

	def fit(self, X, y):
		_y = np.where(y <= 0, -1, 1)
		n_samples, n_features = X.shape
		self.w = np.zeros(n_features)
		self.b = 0

		for _ in range(self.n_iters):
			for i, xi in enumerate(X):
				descent = _y[i] * (np.dot(xi, self.w) - self.b) >= 1
				if descent:
					self.w -= self.lr * (2 * self.lambda_param * self.w)
				else:
					self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(xi, _y[i]))
					self.b -= self.lr * _y[i]

	def predict(self, X):
		linear_output = np.dot(X, self.w) - self.b
		return np.sign(linear_output)