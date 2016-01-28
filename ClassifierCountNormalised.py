import numpy as np
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

def count_state(npArray):
    vehicle_count = 0 #1
    idle_count = 0    #2
    walking_count = 0 #3

    for n in npArray:
        if n == 1:
            vehicle_count += 1
        elif n == 2:
            idle_count += 1
        elif n == 3:
            walking_count += 1

    return {"vehicle_count": vehicle_count, "idle_count" : idle_count, "walking_count":walking_count}

#load dataset
dataset_train = np.loadtxt("training.csv",delimiter=",", skiprows=1)
X = dataset_train[:,0:3]
X_normalised = preprocessing.normalize(X,norm="l2")

#ground truth
y = dataset_train[:,3]

n_neighbors = 3


for i in ["idle.csv","walking.csv","vehicle.csv"]:
    for n_neighbors in range (1,11):
        print "#########################" + i +  " k= " + str(n_neighbors) + ", normalised #########################"
        dataset_test = np.loadtxt(i,delimiter=",",skiprows=1)
        test = dataset_test[:,0:3]
        test_normalised = preprocessing.normalize(test, norm="l2")
        #print test

        ground_truth = dataset_test[:,3]

        # we create an instance of Neighbours Classifier and fit the data.
        knn = neighbors.KNeighborsClassifier(n_neighbors)
        knn.fit(X_normalised, y)

        predicted_result = knn.predict(test_normalised)

        print confusion_matrix(predicted_result,ground_truth)

        result = count_state(predicted_result)
        print "ground truth: "+i, result
