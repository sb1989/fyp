import pandas as pd
import datetime
import os
from sklearn.metrics import confusion_matrix
from sklearn import neighbors
import  math

PATH = "C:\\Users\\SHOUBI\\PycharmProjects\\fyp\\training_datasets\\"
FILENAME = "training.csv"
FILENAME_MEAN = "training_mean.csv"
def slicing(df):
#df = pd.read_csv('datasets/acc_raw_data_trial.csv')
    print type(df)
    signal_timestamp = df["timestamp"]
    signal_x = df["x"]
    signal_y = df["y"]
    signal_z = df["z"]
    #add new column to existing dataframe
    df["window id"] = ""

    WINDOW_DURATION = 10000
    # initialize loop variables
    window_counter = 1;
    prev_window_end = 0;
    window_size = 0;

    signal_timestamp = signal_timestamp - signal_timestamp[0]

    #print signal_timestamp

    #divide the time series into windows of 10 seconds
    window_count_estimated  = math.floor((signal_timestamp[len(signal_timestamp)-1] - signal_timestamp[0])/WINDOW_DURATION)
    #print window_count_estimated

    #determine each data belong to which windows
    for index in range (len(signal_timestamp)):
     #check if it's the end of current window
        if(index+1 < len(signal_timestamp)):
            if (signal_timestamp[index +1] > (window_counter+1)*WINDOW_DURATION):
                window_counter = window_counter + 1
                df.loc[index, "window id"] = window_counter
            else :
                df.loc[index, "window id"] = window_counter
        else:
             df.loc[index, "window id"] = window_counter

    #compute_mean(df, window_counter)
    #print "processed_data"+ datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +".csv"
    #path = "C:\\Users\\SHOUBI\\PycharmProjects\\fyp\\datasets\\"
    #df.to_csv(os.path.join(path, "processed_data_"+ datetime.datetime.now().strftime("%Y-%m-%d%H%M%S") +".csv"), sep=',',index = False)

    return df
    #return ds


def compute_mean(df):
    result = pd.DataFrame()
    result ["window id"] = ""
    result ["acc mean x"] = ""
    result ["acc mean y"] = ""
    result ["acc mean z"] = ""
    grouped = df.groupby(["window id"])
    for k, group in grouped:
        #print group
        lengthOfGroupK = len(group.loc[:,"timestamp"])
        result.loc[k,"window id"] = k
        result.loc[k,"acc mean x"] = group.loc[:,"x"].sum() / lengthOfGroupK
        result.loc[k,"acc mean y"] = group.loc[:,"y"].sum() / lengthOfGroupK
        result.loc[k,"acc mean z"] = group.loc[:,"z"].sum() / lengthOfGroupK
        #print result.loc[k, "acc mean x"] =
        #print group.loc[:,"x"].sum()
        print k
    print result
    return result.ix[:,[1,2,3]]
def main():
    #load training dataset
    dataset_train = pd.read_csv("training_datasets/training_mean.csv",delimiter=",")

    X = dataset_train.ix[:,[0,1,2]]
    X = X.as_matrix()
    #print X
    #print X
    #ground truth
    y = dataset_train.ix[:,[3]]
    y = y.as_matrix()
    y = y.ravel()
    #print y
    n_neighbors = 9
    #create an instance of Neighbours Classifier and fit the data.
    knn = neighbors.KNeighborsClassifier(n_neighbors)
    knn.fit(X, y)

    #load test dataset
    dataset_test = pd.read_csv("datasets/acc_raw_data_trial.csv",delimiter=",",skiprows=0)
    #print dataset_test
    dataset_test = dataset_test.ix[:,[0,1,2,3]]
    #print dataset_test
    #test = test.as_matrix()
    dataset_test = slicing(dataset_test)
    #print dataset_test

    dataset_test =  compute_mean(dataset_test)
    #print type(dataset_test)
    #print dataset_test
    dataset_test = dataset_test.as_matrix()
    #print dataset_test

    #output result into csv
    resultCSVOutput = pd.DataFrame()
    resultCSVOutput["window id"] = ""
    resultCSVOutput["detected state"] = ""
    windowid = 1

    labelDict = {1: "VEHICLE", 2: "IDLE",3:"WALKING"}
    for item in dataset_test:

        print knn.predict(item)
        resultCSVOutput.loc[windowid,"window id"] = windowid
        print resultCSVOutput
        t = knn.predict(item)
        t = t[0]
        print t
        resultCSVOutput.loc[windowid,"detected state"] = labelDict.get(t)
        windowid += 1
    print resultCSVOutput
    path = "C:\\Users\\SHOUBI\\PycharmProjects\\fyp\\output_datasets\\"
    resultCSVOutput.to_csv(os.path.join(path, "processed_data_"+ datetime.datetime.now().strftime("%Y-%m-%d%H%M%S") +".csv"), sep=',',index = False)
main()