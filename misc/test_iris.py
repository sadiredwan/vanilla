import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['FF0000', '00FF00', '0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

from knn import KNN

clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
acc = np.sum(predictions == y_test) / len(y_test)
print(acc)

# from pca import PCA

# clf = PCA(2)
# clf.fit(X)
# X_projected = clf.transform(X)
# print('Shape of X:', X.shape)
# print('Shape of transformed X:', X_projected.shape)
# x1 = X_projected[:, 0]
# x2 = X_projected[:, 1]

# plt.scatter(x1, x2, c=y, edgecolor='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis', 3))
# plt.xlabel('pc-1')
# plt.ylabel('pc-2')
# plt.colorbar()
# plt.show()