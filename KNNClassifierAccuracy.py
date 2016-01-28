from sklearn import preprocessing
from sklearn import neighbors
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.axis as ax

PATH = "C:\\Users\\SHOUBI\\PycharmProjects\\fyp\\training_datasets\\"
FILENAME = "training.csv"
FILENAME_MEAN = "training_mean.csv"

def normalized_dataset(p,f):
    #load dataset
    dataset_train = pd.read_csv(p+f,delimiter=",")
    #X = preprocessing.normalize(dataset_train.ix[:,[0,1,2]],norm="l1")
    #X = dataset_train.ix[:,[0,1,2]]
    X = dataset_train.ix[:,[0,2]]
    #print X
    #scale the data to [0,1] range
    X = X.as_matrix()
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X)
    #print X_train_minmax
    y = dataset_train.ix[:,[3]]
    y= y.as_matrix()
    y = y.ravel()
    # define k odd value from 1 to 30
    k_range = range(1,31)
    k_score = []

    for n_neighbors in k_range:

        knn = neighbors.KNeighborsClassifier(n_neighbors)
        knn.fit(X_train_minmax, y)
        accuracy_score = cross_val_score(knn,X_train_minmax,y,cv=10,scoring='accuracy')
        k_score.append(accuracy_score.mean())
    #plt.figtext("Original Dataset")
    # plt.plot(k_range, k_score)
    # plt.xlabel('Value of K for KNN')
    # plt.ylabel('Cross-Validated Accuracy')
    # plt.show()

    print "normalized: ", k_score

def mean_dataset(p,f):
    #load training dataset
    dataset_train = pd.read_csv(p+f,delimiter=",", skiprows=1)

    #X = dataset_train.ix[:,[2,3,4]]
    X = dataset_train.ix[:,[1,2]]
    X = X.as_matrix()
    #ground truth
    y = dataset_train.ix[:,[3]]
    y = y.as_matrix()
    y = y.ravel()

    # define k odd value from 1 to 30
    k_range = range(1,31)
    k_score = []
    for n_neighbors in k_range:
        #create an instance of Neighbours Classifier and fit the data.
        knn = neighbors.KNeighborsClassifier(n_neighbors)
        knn.fit(X, y)
        accuracy_score = cross_val_score(knn, X,y,cv=10,scoring='accuracy')
        k_score.append(accuracy_score.mean())
    # plt.plot(k_range, k_score)
    # plt.xlabel('Value of K for KNN')
    # plt.ylabel('Cross-Validated Accuracy')

    print "mean: ", k_score

def plot_confusion_matrix(cm, title='', cmap=plt.cm.Blues):
    #print cm
    #display vehicle, idle, walking accuracy respectively
    #display overall accuracy
    print type(cm)
   # plt.figure(index
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.figure("")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = [0,1,2]
    target_name = ["driving","idling","walking"]


    plt.xticks(tick_marks,target_name,rotation=45)

    plt.yticks(tick_marks,target_name,rotation=45)
    print len(cm[0])

    for i in range(0,3):
        for j in range(0,3):
         plt.text(i,j,str(cm[i,j]))
    plt.tight_layout()
    plt.ylabel("Actual Value")
    plt.xlabel("Predicted Outcome")

def oringal_dataset(p,f):
    #load dataset
    dataset_train = pd.read_csv(p+f,delimiter=",")
    #X = dataset_train.ix[:,[0,1,2]]
    X = dataset_train.ix[:,[1,2]]
    X = X.as_matrix()

    y = dataset_train.ix[:,[3]]
    y= y.as_matrix()
    y = y.ravel()

    # k_range = range(1,2)
    # #10 fold
    # sss = StratifiedShuffleSplit(y,10)
    # print len(sss)
    # for n_neighbors in k_range:
    #     cm_total_per_sss = np.zeros((3,3))
    #     for train_index, test_index in sss:
    #        #print("TRAIN:", train_index, "TEST:", test_index)
    #        X_train, X_test = X[train_index], X[test_index]
    #        y_train, y_test = y[train_index], y[test_index]
    #        knn = neighbors.KNeighborsClassifier(n_neighbors)
    #        knn.fit(X_train,y_train)
    #        y_pred = knn.predict(X_test)
    #
    #        #cm for each fold
    #        cm = confusion_matrix(y_test, y_pred)
    #        #print cm
    #        cm_total_per_sss = np.matrix(cm) + cm_total_per_sss
    #        #print cm
    #
    #     #print cm_total_per_sss
    #     plot_confusion_matrix(n_neighbors,cm_total_per_sss.astype(int)/10)


    #define k odd value from 1 to 30
    k_range = range(1,31)

    k_score = []

    for n_neighbors in k_range:
        knn = neighbors.KNeighborsClassifier(n_neighbors)
        knn.fit(X, y)
        accuracy_score = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
        #print accuracy_score
        k_score.append(accuracy_score.mean())
    print "original: ", k_score

def mean_dataset_confusion_matrix(p,f):
    #load training dataset
    dataset_train = pd.read_csv(p+f,delimiter=",", skiprows=1)

    #X = dataset_train.ix[:,[2,3,4]]
    X = dataset_train.ix[:,[0,1,2]]
    X = X.as_matrix()
    #ground truth
    y = dataset_train.ix[:,[3]]
    y = y.as_matrix()
    y = y.ravel()

    # define k odd value from 1 to 30
    k_range = range(1,11)
    # k_score = []
    # for n_neighbors in k_range:
    #     #create an instance of Neighbours Classifier and fit the data.
    n_neighbors = 3
    # knn = neighbors.KNeighborsClassifier(n_neighbors)
    # knn.fit(X, y)
    #for n_neighbors in range(1,10):
    sss = StratifiedKFold(y,10)
    #print len(sss)
    for n_neighbors in k_range:
        cm_total_per_sss = np.zeros((3,3))
        for train_index, test_index in sss:
        #        #print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                knn = neighbors.KNeighborsClassifier(n_neighbors)
                knn.fit(X_train,y_train)
                y_pred = knn.predict(X_test)
                #print "xxxxxxxxxxxxxxx", n_neighbors
        #        #cm for each fold
                cm = confusion_matrix(y_test, y_pred)
                #print cm
                cm_total_per_sss = cm + cm_total_per_sss
        print cm_total_per_sss
#
#     #print cm_total_per_sss
    #plot_confusion_matrix(cm_total_per_sss.astype(int))
    # plt.plot(k_range, k_score)
    # plt.xlabel('Value of K for KNN')
    # plt.ylabel('Cross-Validated Accuracy')

    #print "mean: ", k_score
#oringal_dataset(PATH,FILENAME)
#mean_dataset(PATH,FILENAME_MEAN)
#normalized_dataset(PATH,FILENAME)
mean_dataset_confusion_matrix(PATH,FILENAME_MEAN)
plt.show()













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