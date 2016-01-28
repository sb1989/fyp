import pandas as pd
from sklearn import neighbors
from sklearn.metrics import confusion_matrix

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
df = pd.read_csv("training.csv", delimiter=",", skiprows=0)
#we create an instance of Neighbours Classifier and fit the data.
n_neighbors = 3
X  = df.ix[:, [0, 1, 2]]
X = X.as_matrix()
print X
y  = df.ix[:, [3]]
y = y.as_matrix()
y = y.ravel()
print y
knn = neighbors.KNeighborsClassifier(n_neighbors)
knn.fit(X, y)

#separate dataset into windows of 10 second
grouped = df.groupby(['window_id'])
tempDF = ""
for k, group in grouped:
    #copy all data where window id = k into tempDataFrame
    tempDF = grouped.get_group(k)
    print tempDF
    #remove all data from dataframe where window id = k
    querystatement = 'window_id != ' + str(k)
    df = df.query(querystatement)
    ground_truth = tempDF.ix[:,[3]]
    print ground_truth
    print df
    print tempDF.ix[:,[0,1,2]]
    tempDF = tempDF.ix[:,[0,1,2]].as_matrix()
    print tempDF
    print type(tempDF)
    predicted_result = knn.predict(tempDF)

    print predicted_result
    cm = confusion_matrix(predicted_result,ground_truth)
    print cm
    #add tempDataFrame back to df
    df.append(tempDF)
    print df
#print dataset_test
#grouped = dataset_test.groupby(['window_id'])

#for k, group in grouped:
         # print k
         # print group


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

#
#         predicted_result = knn.predict(test)
#
#         #cm = confusion_matrix(predicted_result,ground_truth)
#         #print cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print confusion_matrix(predicted_result,ground_truth)
#         result = count_state(predicted_result)
#         print "ground truth: "+i, result
#
#     # # Plot also the training points
#     # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
#     # plt.xlim(xx.min(), xx.max())
#     # plt.ylim(yy.min(), yy.max())
#     # plt.title("3-Class classification (k = %i, weights = '%s')"% (n_neighbors,"uniform"))
#     #
#     # plt.show()