import os
import datetime
import pandas as pd
from sklearn import neighbors
import math

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

    return {"VEHICLE": vehicle_count, "IDLE" : idle_count, "WALKING":walking_count}

def classify(df):
    print type(df)
    #load dataset
    dataset_train = pd.read_csv("training.csv",delimiter=",")
    X = dataset_train.ix[:,[0,1,2]]
    X = X.as_matrix()
    print X
    print type(X)
    #X_normalised = preprocessing.normalize(X,norm="l2")
    #ground truth
    y = dataset_train.ix[:,[3]]
    #print y.transpose()
    y = y.as_matrix()
    #y = y.transpose()
    y = y.ravel()
    print y
    print type(y)
    #y = y.reshape((len(y),))
    n_neighbors = 3
    #create an instance of Neighbours Classifier and fit the data.
    knn = neighbors.KNeighborsClassifier(n_neighbors)
    knn.fit(X, y)
    print df.ix[:,[1,2,3]]
    return knn.predict(df.ix[:,[1,2,3]])
#def exportResult(windowid, result):

def main():
    resultCSVOutput = pd.DataFrame()
    resultCSVOutput["window id"] = ""
    resultCSVOutput["detected state"] = ""
    # step 1: load the test dataset and preprocess
    dataset_test = pd.read_csv("datasets/acc_raw_data_trial.csv",delimiter=",")
    #print dataset_test
   # print type(dataset_test)
    test = dataset_test.ix[:,[0,1,2,3]]
    #print type (test)
    dataset_test = slicing(test)
    print dataset_test
    grouped = dataset_test.groupby(['window id'])
    for k, group in grouped:
        #classification using knn
        result =  classify(group)
        print result
        result = count_state(result)
        resultCSVOutput.loc[k,"window id"] = k
        resultCSVOutput.loc[k,"detected state"] = max(result,key = result.get)
        print resultCSVOutput
        #print max(count_state(result))
    path = "C:\\Users\\SHOUBI\\PycharmProjects\\fyp\\output_datasets\\"
    resultCSVOutput.to_csv(os.path.join(path, "processed_data_"+ datetime.datetime.now().strftime("%Y-%m-%d%H%M%S") +".csv"), sep=',',index = False)
main()