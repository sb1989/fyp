import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import neighbors

def compute_mean(df, window_counter):
    grouped = df.groupby(['window_id'])

    outputCSV = pd.DataFrame()
    #outputCSV["window id"] =""
    # ouptuCSV["rel_time_sec"]=""
    #outputCSV["n"]=""
    outputCSV["mean acc x"]=""
    # ouptuCSV["var_acc_x"]=""
    # ouptuCSV["pp_acc_x"]=""
    outputCSV["mean acc y"]=""
    # ouptuCSV["var_acc_y"]=""
    # ouptuCSV["pp_acc_y"]=""
    outputCSV["mean acc z"]=""
    #print outputCSV
    #outputCSV.loc[1,"window id"] = "123"
    #print outputCSV
    # for k, group in grouped:
    #     print k
    #     print group
    for i in range(1,window_counter):
        #outputCSV.loc[i,"window id"] = i
        #outputCSV.loc[i,"n"] = len(grouped.get_group(i)["acc_x"])
        outputCSV.loc[i,"mean acc x"] = grouped.get_group(i)["acc_x"].sum() / len(grouped.get_group(i)["acc_x"])
        outputCSV.loc[i,"mean acc y"] = grouped.get_group(i)["acc_y"].sum() / len(grouped.get_group(i)["acc_y"])
        outputCSV.loc[i,"mean acc z"] = grouped.get_group(i)["acc_z"].sum() / len(grouped.get_group(i)["acc_z"])
    return outputCSV
    #outputCSV.to_csv("asdsa.csv")




#load dataset
dataset_train = pd.read_csv("training_mean.csv",delimiter=",", skiprows=1)

outputCSV = pd.DataFrame()
X = dataset_train.ix[:,[2,3,4]]
X = X.as_matrix()
print X
#print X
#ground truth
y = dataset_train.ix[:,[5]]
y = y.as_matrix()
y = y.ravel()
print y

n_neighbors = 1
for i in ["idle.csv","walking.csv","vehicle.csv"]:
    for n_neighbors in range (1,11):

        print "#########################" + i + " k= " + str(n_neighbors) + ", mean #########################"
        dataset_test = pd.read_csv(i,delimiter=",",skiprows=0)
        #print dataset_test
        test = dataset_test.ix[:,[0,1,2,3,4]]
        #test = test.as_matrix()

        #print dataset_test
        #test_normalised = preprocessing.normalize(test, norm="l2")
        #print test
        test = compute_mean(test, 2)
        print type(test)
        print test
        #ground_truth = dataset_test[:,3]
        test = test.as_matrix()
        # we create an instance of Neighbours Classifier and fit the data.
        knn = neighbors.KNeighborsClassifier(n_neighbors)
        knn.fit(X, y)

        predicted_result = knn.predict(test)

        print predicted_result
        #print confusion_matrix(predicted_result,ground_truth)

        # result = count_state(predicted_result)
        # print "ground truth: "+i, result


