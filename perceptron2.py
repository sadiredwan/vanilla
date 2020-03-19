import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x)

class NeuralNetwork:
	def __init__(self, inputs):
		self.inputs = inputs
		self.l = len(self.inputs)
		self.li = len(self.inputs[0])

		self.weights_input = np.random.random((self.li, self.l))
		self.weights_hidden = np.random.random((self.l, 1))


	def train(self, training_inputs, training_outputs, training_iterations):
		for iteration in range(training_iterations):
			input_layer = training_inputs
			hidden_layer1 = sigmoid(np.dot(input_layer, self.weights_input))
			output_layer = sigmoid(np.dot(hidden_layer1, self.weights_hidden))

			#backpropagation
			cost_hidden_layer1 = training_outputs - output_layer
			grad_hidden_layer1 = np.multiply(cost_hidden_layer1, sigmoid_derivative(output_layer))

			cost_input_layer = np.dot(grad_hidden_layer1, self.weights_hidden.T)
			grad_input_layer = np.multiply(cost_input_layer, sigmoid_derivative(hidden_layer1))

			self.weights_input += np.dot(input_layer.T, grad_input_layer)
			self.weights_hidden += np.dot(hidden_layer1.T, grad_hidden_layer1)

	def predict(self, inputs):
		output_of_input_layer = sigmoid(np.dot(inputs, self.weights_input))
		output_of_hidden_layer1 = sigmoid(np.dot(output_of_input_layer, self.weights_hidden))
		return output_of_hidden_layer1
			
if __name__ == "__main__":
	training_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])

	training_outputs = np.array([[0,1,1,0]]).T

	neural_network = NeuralNetwork(training_inputs)

	# print("Random starting weights: ")
	# print(neural_network.weights_hidden)

	neural_network.train(training_inputs, training_outputs, 10000)

	# print("weights after training: ")
	# print("{}\n".format(neural_network.weights_hidden))

	print(neural_network.predict(np.array([1, 0, 0])))