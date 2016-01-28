import numpy as np
from sklearn import neighbors
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

#load dataset
dataset = np.loadtxt("training.csv",delimiter=",", skiprows=1)
dataset_test = np.loadtxt("walking.csv",delimiter=",", skiprows=1)

test = dataset_test[:,0:2]
#print test
X = dataset[:,0:2]
print X
#print  X
y = dataset[:,3]
print y
#print y
n_neighbors = 1

# we create an instance of Neighbours Classifier and fit the data.
knn = neighbors.KNeighborsClassifier(n_neighbors)
knn.fit(X, y)


cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
h = .02
x_min = X[:, 0].min() - 1
#print x_min
x_max = X[:, 0].max() + 1
#print x_max
y_min = X[:, 1].min() - 1
#print y_min
y_max = X[:, 1].max() + 1
#print y_max
#z_min = X[:, 2].min() - 1
#z_max = X[:, 2].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))


Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.ylabel("accelerometer y")
plt.xlabel("accelerometer x")

plt.colorbar()

#plotting the test data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)


plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, "distance"))
plt.show()


#print knn.predict(test)

# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title("3-Class classification (k = %i, weights = '%s')"% (n_neighbors,"uniform"))
#
# plt.show()