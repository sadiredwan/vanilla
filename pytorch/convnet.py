import os
import cv2
import numpy as np
from tqdm import tqdm

REBUILD_DATA = True


class DvC():
	IMG_SIZE = 64
	CATS = "PetImages/Cat"
	DOGS = "PetImages/Dog"
	LABELS = {CATS: 0, DOGS: 1}

	training_data = []
	catcount = 0
	dogcount = 0

	def make_training_data(self):
		for label in self.LABELS:
			# print(label)
			for file in tqdm(os.listdir(label)):
				try:
					path = os.path.join(label, file)
					img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
					img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
					self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

					if label == self.CATS:
						self.catcount += 1
					elif label == self.DOGS:
						self.dogcount += 1
				except Exception as e:
					pass

		np.random.shuffle(self.training_data)
		np.save("training_data.npy", self.training_data)
		print("Cats: ", self.catcount)
		print("DOGS: ", self.dogcount)


import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 32, 7) # input is 1 image, 32 output channels, 7x7 kernel/window
		self.conv2 = nn.Conv2d(32, 64, 7)
		self.conv3 = nn.Conv2d(64, 128, 7)

		x = torch.randn(64, 64).view(-1, 1, 64, 64)
		self._to_linear = None
		self.convs(x)

		self.fc1 = nn.Linear(self._to_linear, 512)
		self.fc2 = nn.Linear(512, 2)

	def convs(self, x):
		x = Func.max_pool2d(Func.relu(self.conv1(x)), (2, 2))
		x = Func.max_pool2d(Func.relu(self.conv2(x)), (2, 2))
		x = Func.max_pool2d(Func.relu(self.conv3(x)), (2, 2))

		print(x[0].shape)
		if self._to_linear is None:
			self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
			return x

	def forward(self, x):
		x = self.convs(x)
		x = x.view(-1, self._to_linear)
		x = Func.relu(self.fc1(x))
		x = self.fc2(x)
		return Func.softmax(x, dim=1)


if __name__ == '__main__':
	
	if REBUILD_DATA:
		dvc = DvC()
		dvc.make_training_data()

	net = Net()

	optimizer = optim.Adam(net.parameters(), lr=0.001)
	loss_function = nn.MSELoss()

	# training_data =	np.load(training_data.npy)

	# X = torch.Tensor([i[0] for i in training_data]).view(-1, 64, 64)
	# X = X/255.0
	# y = torch.Tensor([i[1] for i in training_data])
