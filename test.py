# from data_preprocessing.dataloader import loading_data
# import argparse

# parser = argparse.ArgumentParser()

# parser.add_argument('--padding', type=str, 
#                     default='mean', help='choose one of them : no, max, mean')
# parser.add_argument('--timespan', type=int, 
#                     default=1000, help='choose of the number of timespan between data points(1000 = 1sec, 60000 = 1min)')
# parser.add_argument('--min_seq', type=int, 
#                     default=10, help='choose of the minimum number of data points in a example')
# parser.add_argument('--min_samples', type=int, default=20, 
#                     help='choose of the minimum number of samples in each label')
# parser.add_argument('--selected_dataset', default='lapras', type=str,
#                     help='Dataset of choice: lapras, casas, opportunity, aras_a, aras_b')
# parser.add_argument('--aug_method', type=str, default='AddNoise', help='choose the data augmentation method')
# parser.add_argument('--aug_wise', type=str, default='None', help='choose the data augmentation wise')

# parser.add_argument('--test_ratio', type=float, default=0.3, help='choose the number of test ratio')

# args = parser.parse_args(args=[])
# data_type = args.selected_dataset



# # ARAS_A
# #num_classes_AA, datalist_AA, labellist_AA = loading_data('aras_a', args)
# #print(datalist_AA.shape)

# # ARAS_B
# #num_classes_AB, datalist_AB, labellist_AB = loading_data('aras_b', args)
# #print(datalist_AB.shape)

# #list_z = [0,1,1,1,0,0,1]
# #temp_list = list_z.copy
# #for i in list_z:
# #    if i == 0:
# #        list_z.remove(i)

# #print(list_z)

# from sklearn.svm import OneClassSVM
# X = [[0], [0.44], [0.45], [0.46], [1]]
# clf = OneClassSVM(gamma='auto').fit(X)
# print(clf.predict(X))#array([-1,  1,  1,  1, -1])
# print(clf.score_samples(X))#array([1.7798..., 2.0547..., 2.0556..., 2.0561..., 1.7332...])

import torch

torch.manual_seed(1)

x = torch.randn((3, 3, 3))
y = torch.sigmoid(x)
print(x.shape)
print(y.min(), y.max(), y)

# Expected output
# (tensor(0.1667), tensor(0.9364))