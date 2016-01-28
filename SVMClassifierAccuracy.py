from sklearn import svm
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd

PATH = "C:\\Users\\SHOUBI\\PycharmProjects\\fyp\\training_datasets\\"
FILENAME = "training.csv"
FILENAME_MEAN = "training_mean.csv"

def normalized_dataset(p,f):
    #load dataset
    dataset_train = pd.read_csv(p+f,delimiter=",")
    #X = preprocessing.normalize(dataset_train.ix[:,[0,1,2]],norm="l1")
    X = dataset_train.ix[:,[0,1,2]]
    #print X
    #scale the data to [0,1] range
    X = X.as_matrix()
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X)
    print X_train_minmax
    #print X_train_minmax
    y = dataset_train.ix[:,[3]]
    y= y.as_matrix()
    y = y.ravel()

     # define k odd value from 1 to 30
    k_score = []
    clf = svm.SVC()
    clf.fit(X_train_minmax,y)
    accuracy_score = cross_val_score(clf,X_train_minmax,y,cv=10,scoring='accuracy')
    k_score.append(accuracy_score.mean())
    #plt.figtext("Original Dataset")
    print k_score

def mean_dataset(p,f):
    #load training dataset
    dataset_train = pd.read_csv(p+f,delimiter=",", skiprows=1)

    X = dataset_train.ix[:,[0,1,2]]
    X = X.as_matrix()
    #ground truth
    y = dataset_train.ix[:,[3]]
    y = y.as_matrix()
    y = y.ravel()

    # define k odd value from 1 to 30
    k_score = []
    clf = svm.SVC()
    clf.fit(X,y)
    accuracy_score = cross_val_score(clf,X,y,cv=10,scoring='accuracy')
    k_score.append(accuracy_score.mean())
    #plt.figtext("Original Dataset")
    print k_score

def oringal_dataset(p,f):
    #load dataset
    dataset_train = pd.read_csv(p+f,delimiter=",")
    X = dataset_train.ix[:,[0,1,2]]
    X = X.as_matrix()

    y = dataset_train.ix[:,[3]]
    y= y.as_matrix()
    y = y.ravel()

    k_score = []

    clf = svm.SVC()
    clf.fit(X,y)
    accuracy_score = cross_val_score(clf,X,y,cv=10,scoring='accuracy')
    k_score.append(accuracy_score.mean())
    #plt.figtext("Original Dataset")
    print k_score
oringal_dataset(PATH,FILENAME)
mean_dataset(PATH,FILENAME_MEAN)
normalized_dataset(PATH,FILENAME)


# for i in ["idle.csv","walking.csv","vehicle.csv"]:
#     for n_neighbors in range (1,11):
#         print "#########################" + i + " k= " + str(n_neighbors) + "#########################"
#         dataset_test = np.loadtxt(i,delimiter=",",skiprows=1)
#
#         test = dataset_test[:,0:3]
#         #test_normalised = preprocessing.normalize(test, norm="l2")
#         #print test
#         #print test
#         #print type(test)
#         ground_truth = dataset_test[:,3]
#         #print ground_truth
#         #print type(ground_truth)
#         # we create an instance of Neighbours Classifier and fit the data.
#         knn = neighbors.KNeighborsClassifier(n_neighbors)
#         knn.fit(X, y)
#
#         predicted_result = knn.predict(test)
#
#         #cm = confusion_matrix(predicted_result,ground_truth)
#         #print cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print confusion_matrix(predicted_result,ground_truth)
#         result = count_state(predicted_result)
#         print "ground truth: "+i, result

    # # Plot also the training points
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    # plt.title("3-Class classification (k = %i, weights = '%s')"% (n_neighbors,"uniform"))
    #
    # plt.show()