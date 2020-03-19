import numpy as np

class NeuralNetwork():
	def __init__(self):
		np.random.seed(1)
		# bias from -1 to 1, mean 0
		self.weights = 2 * np.random.random((3, 1)) - 1

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivative(self, x):
		return x * (1 - x)

	def train(self, training_inputs, training_outputs, training_iterations):
		for iteration in range(training_iterations):
			output = self.predict(training_inputs)
			cost = training_outputs - output
			grad = np.dot(training_inputs.T, cost * self.sigmoid_derivative(output))
			self.weights += grad

	def predict(self, inputs):
		inputs = inputs.astype(float)
		output = self.sigmoid(np.dot(inputs, self.weights))
		return output


if __name__ == "__main__":
	neural_network = NeuralNetwork()

	# print("Random starting weights: ")
	# print(neural_network.weights)

	training_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])

	training_outputs = np.array([[0,1,1,0]]).T
	neural_network.train(training_inputs, training_outputs, 10000)

	# print("weights after training: ")
	# print("{}\n".format(neural_network.weights))
	
	print(neural_network.predict(np.array([1, 0, 0])))