import math
import pandas as pd
import datetime
import os
#Constants
WINDOW_DURATION = 10000 # miliseconds
#GRAVITY_EARTH = 9.80665


def compute_mean(df, window_counter):
    grouped = df.groupby(['window id'])

    outputCSV = pd.DataFrame()
    outputCSV["window id"] =""
    # ouptuCSV["rel_time_sec"]=""
    outputCSV["n"]=""
    outputCSV["mean acc x"]=""
    # ouptuCSV["var_acc_x"]=""
    # ouptuCSV["pp_acc_x"]=""
    outputCSV["mean acc y"]=""
    # ouptuCSV["var_acc_y"]=""
    # ouptuCSV["pp_acc_y"]=""
    outputCSV["mean acc z"]=""
    #print outputCSV
    #outputCSV.loc[1,"window id"] = "123"
    print outputCSV
    # for k, group in grouped:
    #     print k
    #     print group
    for i in range(1,window_counter):
        outputCSV.loc[i,"window id"] = i
        outputCSV.loc[i,"n"] = len(grouped.get_group(i)["x"])
        outputCSV.loc[i,"mean acc x"] = grouped.get_group(i)["x"].sum() / len(grouped.get_group(i)["x"])
        outputCSV.loc[i,"mean acc y"] = grouped.get_group(i)["y"].sum() / len(grouped.get_group(i)["y"])
        outputCSV.loc[i,"mean acc z"] = grouped.get_group(i)["z"].sum() / len(grouped.get_group(i)["z"])
    print outputCSV
# return s/n

# ouptuCSV["var_acc_z"]=""
# ouptuCSV["pp_acc_z"]=""
# ouptuCSV["mean_acc_mag"]=""
# ouptuCSV["var_acc_mag"]=""
# ouptuCSV["pp_acc_mag"]=""
# ouptuCSV["peak_freq_along_gravity"]=""
# ouptuCSV["peak_freq_mag"]=""
# ouptuCSV["ground_truth"]=""
# ouptuCSV["detected_context"]=""

def main():
    #read csv data
    df = pd.read_csv('datasets/acc_raw_data_trial.csv')
    signal_timestamp = df["timestamp"]
    signal_x = df["x"]
    signal_y = df["y"]
    signal_z = df["z"]
    #add new column to existing dataframe
    df["window id"] = ""


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

    compute_mean(df, window_counter)
    #print "processed_data"+ datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +".csv"
    #path = "C:\\Users\\SHOUBI\\PycharmProjects\\fyp\\datasets\\"
    #df.to_csv(os.path.join(path, "processed_data_"+ datetime.datetime.now().strftime("%Y-%m-%d%H%M%S") +".csv"), sep=',',index = False)

main()