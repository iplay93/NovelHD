from glob import glob
import numpy as np
import pandas as pd

class TSDataSet:
    def __init__(self,data, label, length):
        self.data = data
        self.label = int(label)
        self.length= int(length)
        


# ARAS data format : (for each second) sensor type+context name,..., activity label1, activity label2 : [1-27]/ file name = day 
# Examples(txt) : 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 13 17
# A: [1217, 2217, 1215, 2117, 1517, 1717, 1711, 1111, 1120, 1121, 1115, 1127, 1102, 102, 1202, 302, 402, 1502, 1702, 2102, 1302, 2202, 1902, 2702, 202, 214, 217, 212, 222, 210, 112, 712, 717, 715, 708, 808, 809, 817, 2221, 2212, 1212, 2112, 2622, 2722, 122, 1522, 1527, 2627, 2612, 1012, 2615, 2620, 2601, 2617, 2610, 1210, 1110, 1101, 902, 1002, 2402, 1402, 1422, 1427, 1418, 2525, 1825, 1818, 1812, 2717, 2727, 1722, 1222, 2215, 1214, 1221, 2511, 1511, 2111, 1104, 502, 227, 201, 2712, 1512, 1712, 1707, 707, 818, 1218, 1209, 922, 2222, 1710, 1718, 1211, 602, 1327, 1322, 2522, 1407, 702, 908, 909, 912, 215, 1510, 1509, 1521, 727, 722, 822, 812, 918, 1022, 1017, 1802, 2527, 1727, 1716, 116, 2216, 2214, 114, 1714, 109, 1721, 2121, 121, 115, 802, 1816, 1216, 2218, 2211, 1116, 117, 1715, 1701, 2302, 1516, 1817, 1814, 1810, 2116, 2110, 1112, 1815, 312, 412, 416, 916, 1412, 512, 612, 1827, 101, 1801, 1822, 1709, 1602, 110, 2201, 1109, 1227, 1027, 1015, 1201, 1207, 921, 2210, 1821, 127, 1501, 1010, 1312, 120, 2209, 1807, 107, 1001, 2101, 1614, 1610, 316, 317, 401, 415, 901, 1401, 2701, 1118, 1122, 301, 2118, 207, 1507, 2707, 2207, 1009, 1114, 1011, 1627, 1601, 2227, 1310, 1309, 1301, 1321, 1311, 911, 2311, 718, 716, 216, 915, 2721, 1411, 111, 927, 1315, 209, 221, 2602, 2517, 1103, 2715, 1317, 1325, 2515, 1314, 1125, 2503, 1503, 404, 421, 920, 1720, 1713, 1213, 113, 2213, 1723, 1316, 1708, 2308, 2309, 1518, 322, 422, 2127, 2719, 2312, 1719, 1813, 1220, 2307, 1108, 2512, 522, 517, 617, 917, 1318]
# A: [18, 16, 16, 4, 6, 19, 18, 35, 18, 13, 43, 22, 28, 61, 90, 34, 24, 71, 94, 32, 40, 69, 5, 37, 43, 2, 5, 15, 9, 6, 15, 7, 3, 2, 1, 12, 5, 1, 4, 34, 59, 5, 2, 3, 4, 2, 2, 1, 2, 13, 1, 1, 2, 2, 1, 11, 9, 18, 41, 25, 3, 18, 4, 3, 1, 16, 1, 3, 15, 1, 5, 12, 23, 6, 3, 7, 1, 6, 14, 3, 23, 7, 12, 8, 19, 49, 7, 10, 2, 4, 5, 3, 12, 10, 5, 4, 17, 4, 6, 2, 1, 9, 3, 7, 11, 4, 4, 3, 3, 2, 3, 1, 3, 2, 2, 5, 12, 3, 8, 6, 6, 5, 5, 2, 7, 2, 6, 2, 1, 5, 5, 4, 13, 3, 5, 11, 3, 10, 10, 6, 8, 3, 4, 3, 4, 1, 3, 2, 4, 3, 1, 1, 6, 3, 2, 3, 9, 6, 5, 5, 5, 3, 6, 4, 7, 3, 1, 6, 3, 2, 6, 1, 4, 4, 6, 8, 2, 2, 1, 4, 1, 3, 1, 1, 1, 1, 1, 1, 2, 2, 4, 2, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 1, 2, 2, 1, 7, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 3, 1, 1, 2, 1, 2, 1, 6, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 4, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# B: [1111, 1511, 111, 1115, 1118, 118, 1518, 318, 418, 404, 918, 115, 1318, 1315, 1313, 513, 606, 106, 101, 901, 913, 1316, 1516, 1310, 1018, 1218, 1215, 1212, 1512, 117, 1517, 317, 2525, 2727, 2127, 2121, 121, 202, 1527, 2715, 2701, 701, 801, 808, 1201, 1101, 1117, 127, 2118, 315, 1715, 1804, 1817, 1317, 1321, 1417, 2717, 2101, 1217, 2714, 114, 1814, 1827, 1801, 1501, 2501, 1112, 2711, 2702, 1102, 1127, 2111, 102, 1502, 1103, 1104, 1120, 302, 402, 902, 1802, 1302, 1002, 502, 602, 702, 802, 1202, 2102, 311, 401, 1525, 2515, 518, 515, 112, 1012, 1504, 1211, 1311, 1011, 1818, 1121, 1401, 1702, 1902, 1402, 1015, 1027, 1001, 2112, 411, 1811, 1322, 1304, 517, 915, 717, 815, 1427, 1727, 1701, 2712, 1712, 1227, 304, 1301, 1327, 2302, 917, 1717, 1815, 2415, 2418, 1918, 1418, 2718, 715, 912, 301, 1721, 727, 712, 108, 1708, 1017, 1116, 2716, 316, 104, 120, 920, 718, 1208, 1416, 116, 2211, 921, 2721, 1412, 1513, 413]
# B: [43, 21, 9, 26, 3, 13, 11, 2, 1, 17, 2, 20, 13, 8, 2, 1, 3, 1, 55, 6, 1, 2, 3, 1, 4, 3, 9, 39, 10, 8, 16, 8, 14, 15, 7, 1, 2, 29, 8, 8, 9, 1, 2, 4, 2, 11, 10, 23, 2, 6, 1, 4, 7, 14, 1, 2, 7, 7, 2, 4, 5, 1, 1, 3, 14, 1, 5, 2, 12, 9, 4, 2, 33, 31, 1, 1, 2, 4, 4, 2, 3, 24, 7, 3, 3, 6, 5, 25, 5, 7, 3, 1, 1, 1, 1, 8, 1, 2, 1, 7, 1, 3, 1, 2, 1, 2, 2, 1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 4, 2, 1, 3, 1, 4, 2, 3, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def arasLoader(file_name, timespan, min_seq):

    print("Loading ARAS Dataset--------------------------------------")
    
    # variable initialization
    file_list = [] # store file names
    current_label = [0,0] # current label
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
        temp_df = pd.read_csv(file, sep = ' ', header = None).to_numpy()

        print(file)
        # at least one ADL exist in the file
        if(len(temp_df)>0):
            # for the first row
            current_label[0] = temp_df[0, 20] # 20 column is the label of resident1
            
            current_label[1] = temp_df[0, 21] # 21 column is the label of resident2            
            
            temp_dataset = np.array([temp_df[0,0:19]]) # 0-19 column is the sensors
            current_datalist = temp_df[0,0:19]      
            current_time = 0 
            # for each row 
            for i in range(1, len(temp_df)):
                # for each timespan sec
                if((i-current_time) >= (timespan/1000)):
                    current_time = i
                    # if the same activity continue                
                    if((current_label[0] == temp_df[i, 20]) and (current_label[1] == temp_df[i, 21])):
                        if (current_datalist !=  temp_df[i,0:19]).any():                   
                            temp_dataset = np.concatenate((temp_dataset, [np.array(temp_df[i,0:19])]), axis=0)
                            current_datalist =  temp_df[i,0:19]  
                    # if the activity is finished (new activity arrival)                   
                    else:
                        if(len(temp_dataset)>min_seq):
                            # construct new object(for old activity)
                            if(current_label[0] != temp_df[i, 20]):  # first resident's activity is changed
                                dataset_list.append(TSDataSet(temp_dataset, (current_label[0]), len(temp_dataset)))
                            else: # second resident's activity is changed
                                dataset_list.append(TSDataSet(temp_dataset, (current_label[1]), len(temp_dataset)))
                            # just for show 
                            label_list.append(current_label)      
                                        
                        # new activity append (likely the first row)
                        temp_dataset = np.array([temp_df[i,0:19]])                                   
                        current_label[0] = temp_df[i, 20] 
                        current_label[1] = temp_df[i, 21]

            if(len(temp_dataset)>min_seq):
                # for the last activity
                if(current_label[0] != temp_df[i, 20]):          
                    dataset_list.append(TSDataSet(temp_dataset, (current_label[0]), len(temp_dataset)))
                else:
                    dataset_list.append(TSDataSet(temp_dataset, (current_label[1]), len(temp_dataset)))              
                    
                label_list.append(current_label)
    print("Loading ARAS Dataset Finished--------------------------------------")

    return dataset_list
