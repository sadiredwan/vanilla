import torch
import torch.nn as nn

if __name__ == '__main__':

	X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
	Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

	n_samples, n_features = X.shape
	print(f'#samples: {n_samples}, #features: {n_features}')
	X_test = torch.tensor([5], dtype=torch.float32)

	input_size = n_features
	output_size = n_features
	model = nn.Linear(input_size, output_size)

	print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

	learning_rate = 0.01
	n_iters = 100

	loss = nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

	for epoch in range(n_iters):
		y_predicted = model(X)
		l = loss(Y, y_predicted)
		l.backward()
		optimizer.step()
		optimizer.zero_grad()

		if epoch % 10 == 0:
			[w, b] = model.parameters() # unpack parameters
			print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l)

	print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')