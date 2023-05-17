from glob import glob
import numpy as np
import pandas as pd

class TSDataSet:
    def __init__(self,data, label, length):
        self.data = data
        self.label = int(label)
        self.length= int(length)
        

# Opportunity data format : sensor type+context name,..., activity label(ADL)/ file name = each user(4) * 5
# Examples(txt) : 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 13 17
# the number of examples : 101(Relaxing) - 40, 102(Coffee time)-20, 103(Early morning)-20, 104(Cleanup)-20, 105(Sandwich time)-20
# [1, 3, 2, 5, 4] : activity #
# [40, 20, 20, 20, 20] : # of activity 
def opportunityLoader(file_name, timespan, min_seq):
    print("Loading Opportunity Dataset --------------------------------------")
    # variable initialization
    file_list = [] # store file names
    current_label = 0 # current label
    current_time = 0 # current time

    # return variable (an object list)
    dataset_list = []

    # show how labels are displayed
    label_list = []

    # extract file names
    for x in glob(file_name):
        file_list.append(x)
    # sorting by file name
    file_list.sort()


    # for each file
    for file in file_list :
        temp_df = pd.read_csv(file, sep = ' ', header = None)
        # extract data related to the target ADLs (column :244 => 101~105) and convert to numpy array
        temp_df = temp_df[temp_df[244]>100].to_numpy() 
        print(file)

        # at least one ADL exist in the file
        if(len(temp_df)>0):
            # for the first row
            current_label = temp_df[0, 244] # 244 column is the label
            current_time =  temp_df[0, 0] # 0 column is the timestamp
            temp_dataset = np.array([temp_df[0,1:242]]) # 1-242 column is the sensors       

            # for each row 
            for i in range(1, len(temp_df)):
                # for each timespan sec
                if((temp_df[i, 0]-current_time) >= timespan): 
                    current_time = temp_df[i, 0]
                    # if the same activity continue                
                    if(current_label == temp_df[i, 244]):                           
                        temp_dataset = np.concatenate((temp_dataset, [np.array(temp_df[i,1:242])]), axis=0)
                    # if the activity is finished (new activity arrival)
                    else:
                        # construct new object(for old activity)          
                        dataset_list.append(TSDataSet(temp_dataset, (current_label-100), len(temp_dataset)))
                        # just for show 
                        label_list.append(current_label)          
                                        
                        # new activity append (likely the first row)
                        temp_dataset = np.array([temp_df[i,1:242]])                                   
                        current_label = temp_df[i, 244]

            # for the last activity
            dataset_list.append(TSDataSet(temp_dataset,  (current_label-100), len(temp_dataset)))
            # just for show
            label_list.append(current_label)
    print("Loading Opportunity Dataset Finished--------------------------------------")
    return dataset_list