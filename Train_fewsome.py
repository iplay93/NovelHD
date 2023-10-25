import torch
from models.FewSome_Net import *
import os
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
import torch.optim as optim
#from Test_fewsome import evaluate
import random
import time
from data_preprocessing.dataloader import loading_data
from torch.utils.data import DataLoader, Dataset
import math, random
from sklearn.model_selection import train_test_split
from data_preprocessing.dataloader import count_label_labellist
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn import metrics

from util import calculate_acc, visualization_roc, print_rs
from ood_metrics import auroc, fpr_at_95_tpr

# visualization
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import RocCurveDisplay
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer

test_ratio = 0.2

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_list, label_list, indexes, task):
        super(Load_Dataset, self).__init__()

        self.indexes = indexes
        self.task = task  # training set or test set
        
        X_train = data_list
        y_train = label_list

        # make sure the Channels in second dim
        X_train = np.transpose(X_train,(0, 2, 1))
        # (N, C, T)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]

    def __getitem__(self, index: int, seed = 1, base_ind=-1):
        
        base=False
        data, target = self.x_data[index], int(self.y_data[index])
        #data = torch.stack((data,data,data),0)

        if self.task == 'train':
            np.random.seed(seed)
            ind = np.random.randint(len(self.indexes))
            c = 1

            while (ind == index): #if img2 is the same as img, regenerate ind
                np.random.seed(seed * c)
                ind = np.random.randint(len(self.indexes) )
                c=c+1

            if ind == base_ind:
                base = True #img2 is equal to the anchor

            data2 , target2 = self.x_data[ind], int(self.y_data[ind])
            #data2 = torch.stack((data2,data2,data2),0)
            label = torch.FloatTensor([0])
            
        else:
            data2 = torch.Tensor([1])
            label = target

        return data, data2, label, base        

    def __len__(self):
        return self.len
    
def data_generator(args, datalist, labellist):
    test_ratio = args.test_ratio
    seed =  args.seed 

    # Split train and valid dataset
    train_list, test_list, train_label_list, test_label_list \
    = train_test_split(datalist, labellist, test_size=test_ratio, stratify= labellist, random_state=seed) 


    print(f"Train Data: {len(train_list)} --------------")
    exist_labels, _ = count_label_labellist(train_label_list)

    print(f"Test Data: {len(test_list)} --------------")
    count_label_labellist(test_label_list) 
    
    train_list = torch.tensor(train_list).cuda().cpu()
    train_label_list = torch.tensor(train_label_list).cuda().cpu()

    test_list = torch.tensor(test_list).cuda().cpu()
    test_label_list = torch.tensor(test_label_list).cuda().cpu()

 
    if(args.normal_class != -1): # one-class
        sup_class_idx = [x for x in exist_labels]
        known_class_idx = [args.normal_class]
        novel_class_idx = [item for item in sup_class_idx if item not in set(known_class_idx)]
        
        train_list = train_list[np.where(train_label_list == args.normal_class)]
        train_label_list = train_label_list[np.where(train_label_list == args.normal_class)]

        valid_list = test_list[np.where(test_label_list == args.normal_class)]
        valid_label_list = test_label_list[np.where(test_label_list == args.normal_class)]

        # only use for testing novelty
        test_list = test_list[np.where(test_label_list != args.normal_class)]
        test_label_list = test_label_list[np.where(test_label_list != args.normal_class)]

    else: # multi-class
        sup_class_idx = [x for x in exist_labels]
        random.seed(args.seed)
        known_class_idx = random.sample(sup_class_idx, math.ceil(len(sup_class_idx)/2))

        novel_class_idx = [item for item in sup_class_idx if item not in set(known_class_idx)]
        
        train_list = train_list[np.isin(train_label_list, known_class_idx)]
        train_label_list = train_label_list[np.isin(train_label_list, known_class_idx)]
        valid_list = test_list[np.isin(test_label_list, known_class_idx)]
        valid_label_list =test_label_list[np.isin(test_label_list, known_class_idx)]

        # only use for testing novelty
        test_list = test_list[np.isin(test_label_list, novel_class_idx)]
        test_label_list = test_label_list[np.isin(test_label_list, novel_class_idx)]    


    #normal_class=> 0, abnormal_class  => 1
        
    train_label_list[:] = 0
    valid_label_list[:] = 0
    test_label_list[:] = 1

    # build data loader (N, T, C) -> (N, C, T)
   
      
    #    valid_dataset = Load_Dataset(valid_list, valid_label_list)    
    #test_dataset = Load_Dataset(test_list, test_label_list)    
    replace_list = np.concatenate((valid_list, test_list),axis=0)
    replace_label_list = np.concatenate((valid_label_list, test_label_list),axis=0)

    

    return train_list, train_label_list, replace_list, replace_label_list, novel_class_idx  

def deactivate_batchnorm(m):
    '''
        Deactivate batch normalisation layers
    '''
    if isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, alpha, anchor, device, v=0.0,margin=0.8):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.v = v
        self.alpha = alpha
        self.anchor = anchor
        self.device = device

    def forward(self, output1, vectors, label):
        '''
        Args:
            output1 - feature embedding/representation of current training instance
            vectors - list of feature embeddings/representations of training instances to contrast with output1
            label - value of zero if output1 and all vectors are normal, one if vectors are anomalies
        '''

        euclidean_distance = torch.FloatTensor([0]).to(self.device)

        #get the euclidean distance between output1 and all other vectors
        for i in vectors:
          euclidean_distance += (F.pairwise_distance(output1, i)/ torch.sqrt(torch.Tensor([output1.size()[1]])).to(self.device))


        euclidean_distance += self.alpha*((F.pairwise_distance(output1, self.anchor)) /torch.sqrt(torch.Tensor([output1.size()[1]])).to(self.device) )

        #calculate the margin
        marg = (len(vectors) + self.alpha) * self.margin

        #if v > 0.0, implement soft-boundary
        if self.v > 0.0:
            euclidean_distance = (1/self.v) * euclidean_distance
        #calculate the loss

        loss_contrastive = ((1-label) * torch.pow(euclidean_distance, 2) * 0.5) + ( (label) * torch.pow(torch.max(torch.Tensor([ torch.tensor(0), marg - euclidean_distance])), 2) * 0.5)

        return loss_contrastive

def create_batches(lst, n):
    '''
    Args:
        lst - list of indexes for training instances
        n - batch size
    '''
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def evaluate(anchor, seed, base_ind, ref_dataset, val_dataset, model, dataset_name, normal_class, model_name, indexes, criterion, alpha, num_ref_eval, device):

    model.eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    #create loader for test dataset
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
    outs={} #a dictionary where the key is the reference image and the values is a list of the distances between the reference image and all images in the test set
    ref_images={} #dictionary for feature vectors of reference set
    ind = list(range(0, num_ref_eval))
    np.random.shuffle(ind)
    #loop through the reference images and 1) get the reference image from the dataloader, 2) get the feature vector for the reference image and 3) initialise the values of the 'out' dictionary as a list.
    
    for i in ind:
      img1, _, _, _ = ref_dataset.__getitem__(i)
      if (i == base_ind):
        ref_images['images{}'.format(i)] = anchor
      else:
        ref_images['images{}'.format(i)] = model.forward( img1.to(device).float())

      outs['outputs{}'.format(i)] =[]

    means = []
    minimum_dists=[]
    lst=[]
    labels=[]
    loss_sum =0
    inf_times=[]
    total_times= []
    #loop through images in the dataloader
    
    total_time = 0
    starter.record()
    with torch.no_grad():
        for i, data in enumerate(loader):

            image = data[0][0]
            label = data[2].item()

            labels.append(label)
            total =0
            mini=torch.Tensor([1e50])
            t1 = time.time()
            
            out = model.forward(image.to(device).float()) #get feature vector (representation) for test image
            inf_times.append(time.time() - t1)

            #calculate the distance from the test image to each of the datapoints in the reference set
            for j in range(0, num_ref_eval):
                euclidean_distance = (F.pairwise_distance(out, ref_images['images{}'.format(j)]) / torch.sqrt(torch.Tensor([out.size()[1]])).to(device) ) + (alpha*(F.pairwise_distance(out, anchor) /torch.sqrt(torch.Tensor([out.size()[1]])).to(device) ))

                outs['outputs{}'.format(j)].append(euclidean_distance.item())
                total += euclidean_distance.item()
                if euclidean_distance.detach().item() < mini:
                  mini = euclidean_distance.item()

                loss_sum += criterion(out,[ref_images['images{}'.format(j)]], label).item()

            minimum_dists.append(mini)
            means.append(total/len(indexes))
            total_times.append(time.time()-t1)

            del image
            del out
            del euclidean_distance
            del total
            torch.cuda.empty_cache()
    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    

    total_time = curr_time/len(loader.dataset)

    print("inference time", curr_time, total_time)

    #create dataframe of distances to each feature vector in the reference set for each test feature vector
    cols = ['label','minimum_dists', 'means']
    df = pd.concat([pd.DataFrame(labels, columns = ['label']), pd.DataFrame(minimum_dists, columns = ['minimum_dists']),  pd.DataFrame(means, columns = ['means'])], axis =1)
    for i in range(0, num_ref_eval):
        df= pd.concat([df, pd.DataFrame(outs['outputs{}'.format(i)])], axis =1)
        cols.append('ref{}'.format(i))
    df.columns=cols
    df = df.sort_values(by='minimum_dists', ascending = False).reset_index(drop=True)


    #calculate metrics
    fpr, tpr, thresholds = roc_curve(np.array(df['label']), np.array(df['minimum_dists']))
    
    scores = np.array(df['minimum_dists']).tolist()
    labels = np.array(df['label']).tolist()

    # fp = len(df.loc[(outputs == 1 ) & (df['label'] == 0)])
    # tn = len(df.loc[(outputs== 0) & (df['label'] == 0)])
    # fn = len(df.loc[(outputs == 0) & (df['label'] == 1)])
    # tp = len(df.loc[(outputs == 1) & (df['label'] == 1)])
    # spec = tn / (fp + tn)
    # recall = tp / (tp+fn)
    # acc = (recall + spec) / 2
    # print('AUC: {}'.format(auc_min))
    # print('F1: {}'.format(f1))
    # print('Balanced accuracy: {}'.format(acc))

    auroc, fpr, f1, acc = calculate_acc(labels, scores)

    # fpr, tpr, thresholds = roc_curve(np.array(df['label']),np.array(df['means']))
    # auc = metrics.auc(fpr, tpr)


    #create dataframe of feature vectors for each image in the reference set
    # feat_vecs = pd.DataFrame(ref_images['images0'].detach().cpu().numpy())
    # for j in range(1, num_ref_eval):
    #     feat_vecs = pd.concat([feat_vecs, pd.DataFrame(ref_images['images{}'.format(j)].detach().cpu().numpy())], axis =0)

    # avg_loss = (loss_sum / num_ref_eval )/ val_dataset.__len__()

    return auroc, fpr, f1, acc, scores, labels, total_time




def train(model, lr, weight_decay, train_dataset, val_dataset, epochs, criterion, alpha, model_name, indexes, normal_class, dataset_name, smart_samp, k, eval_epoch, model_type, bs, num_ref_eval, num_ref_dist, device):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    patience = 0
    max_patience = 2 #patience based on train loss
    max_iter = 0
    patience2 = 10 #patience based on evaluation AUC
    best_val_auc = 0
    best_val_auc_min = 0
    best_f1=0
    best_acc=0
    stop_training = False


    start_time = time.time()

    for epoch in range(epochs):
     #   print("Starting epoch " + str(epoch+1))

        model.train()

        loss_sum = 0

        #create batches for epoch
        np.random.seed(epoch)
        np.random.shuffle(ind)
        batches = list(create_batches(ind, bs))

        #iterate through each batch
        for i in range(int(np.ceil(len(ind) / bs))):

            #iterate through each training instance in batch
            for batch_ind,index in enumerate(batches[i]):

                seed = (epoch+1) * (i+1) * (batch_ind+1)
                img1, img2, labels, base = train_dataset.__getitem__(index, seed, base_ind)

                # Forward
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)

                if (index == base_ind):
                  output1 = anchor
                else:
                  output1 = model(img1)

                if (smart_samp == 0) & (k>1):

                  vecs=[]
                  ind2 = ind.copy()
                  np.random.seed(seed)
                  np.random.shuffle(ind2)
                  for j in range(0, k):
                    if (ind2[j] != base_ind) & (index != ind2[j]):
                      output2=model(train_dataset.__getitem__(ind2[j], seed, base_ind)[0].to(device).float())
                      vecs.append(output2)

                elif smart_samp == 0:

                  if (base == True):
                    output2 = anchor
                  else:
                    output2 = model.forward(img2.float())

                  vecs = [output2]

                else:
                  max_eds = [0] * k
                  max_inds = [-1] * k
                  max_ind =-1
                  vectors=[]

                  ind2 = ind.copy()
                  np.random.seed(seed)
                  np.random.shuffle(ind2)
                  for j in range(0, num_ref_dist):

                    if ((num_ref_dist ==1) & (ind2[j] == base_ind)) | ((num_ref_dist ==1) & (ind2[j] == index)):
                        c = 0
                        while ((ind2[j] == base_ind) | (index == ind2[j])):
                            np.random.seed(seed * c)
                            j = np.random.randint(len(ind) )
                            c = c+1

                    if (ind2[j] != base_ind) & (index != ind2[j]):
                      output2=model(train_dataset.__getitem__(ind2[j], seed, base_ind)[0].to(device).float())
                      vectors.append(output2)
                      euclidean_distance = F.pairwise_distance(output1, output2)

                      for b, vec in enumerate(max_eds):
                          if euclidean_distance > vec:
                            max_eds.insert(b, euclidean_distance)
                            max_inds.insert(b, len(vectors)-1)
                            if len(max_eds) > k:
                              max_eds.pop()
                              max_inds.pop()
                            break

                  vecs = []

                  for x in max_inds:
                      with torch.no_grad():
                          vecs.append(vectors[x])

                if batch_ind ==0:
                    loss = criterion(output1,vecs,labels)
                else:
                    loss = loss + criterion(output1,vecs,labels)

            loss_sum+= loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


        train_losses.append((loss_sum / len(ind))) #average loss for each training instance

       # print("Epoch: {}, Train loss: {}".format(epoch+1, train_losses[-1]))


        #if (eval_epoch == 1):
        #    training_time = time.time() - start_time
        #    eval_start_time = time.time()
        #    val_auc, val_loss, val_auc_min, f1, acc,df, ref_vecs, inf_times, total_times = evaluate(anchor, seed, base_ind, train_dataset, val_dataset, model, dataset_name, normal_class, model_name, indexes, criterion, alpha, num_ref_eval, device)
        #    eval_time = time.time() - eval_start_time
        #    print('Validation AUC is {}'.format(val_auc))
        #    print("Epoch: {}, Validation loss: {}".format(epoch+1, val_loss))
        #    if val_auc_min > best_val_auc_min:
        #        best_val_auc = val_auc
        #        best_val_auc_min = val_auc_min
        #        best_epoch = epoch
        #        best_f1 = f1
        #        best_acc = acc
        #        best_df=df
        #        max_iter = 0
        #        training_time_best = (time.time() - start_time) - (eval_time*(epoch+1))

        #        training_time = (time.time() - start_time) - (eval_time*(epoch+1))
        #        write_results(model_name, normal_class, model, df, ref_vecs,num_ref_eval, num_ref_dist, val_auc, epoch, val_auc_min, training_time_best,f1,acc, train_losses, inf_times, total_times)

        #    else:
        #        max_iter+=1

        #    if max_iter == patience2:
        #        break

        #elif args.early_stopping ==1:
        #    if epoch > 1:
        #      decrease = (((train_losses[-3] - train_losses[-2]) / train_losses[-3]) * 100) - (((train_losses[-2] - train_losses[-1]) / train_losses[-2]) * 100)

        #      if decrease <= 0.5:
        #        patience += 1


        #      if (patience==max_patience) | (epoch == epochs-1):
        #          stop_training = True


        #elif (epoch == (epochs -1)) & (eval_epoch == 0):
        #    stop_training = True




        #if stop_training == True:
        #    print("--- %s seconds ---" % (time.time() - start_time))
        #    training_time = time.time() - start_time
        #    val_auc, val_loss, val_auc_min, f1,acc, df, ref_vecs, inf_times, total_times = evaluate(anchor,seed, base_ind, train_dataset, val_dataset, model, dataset_name, normal_class, model_name, indexes, criterion, alpha, num_ref_eval, device)


        #    write_results(model_name, normal_class, model, df, ref_vecs,num_ref_eval, num_ref_dist, val_auc, epoch, val_auc_min, training_time,f1,acc, train_losses, inf_times, total_times)

        #    break

    
    #print("Epoch: {}, Validation loss: {}".format(epoch+1, val_loss))

    print("Finished Training")
    #if eval_epoch == 1:
    #    print("AUC was {} on epoch {}".format(best_val_auc_min, best_epoch+1))
    #    return best_val_auc, best_epoch, best_val_auc_min, training_time_best, best_f1, best_acc,train_losses
    #else:
    #    print("AUC was {} on epoch {}".format(val_auc_min, epoch+1))
    #    return val_auc, epoch, val_auc_min, training_time, f1,acc, train_losses


def write_results(model_name, normal_class, model, df, ref_vecs,num_ref_eval, num_ref_dist, val_auc, epoch, val_auc_min, training_time,f1,acc, train_losses, inf_times, total_times):
    '''
        Write out results to output directories and save model
    '''

    model_name_temp = model_name + '_epoch_' + str(epoch+1) + '_val_auc_' + str(np.round(val_auc, 3)) + '_min_auc_' + str(np.round(val_auc_min, 3))
    for f in os.listdir('./outputs/models/class_'+str(normal_class) + '/'):
      if (model_name in f) :
          os.remove(f'./outputs/models/class_'+str(normal_class) + '/{}'.format(f))
    torch.save(model.state_dict(), './outputs/models/class_'+str(normal_class)+'/' + model_name_temp)


    for f in os.listdir('./outputs/ED/class_'+str(normal_class) + '/'):
      if (model_name in f) :
          os.remove(f'./outputs/ED/class_'+str(normal_class) + '/{}'.format(f))
    df.to_csv('./outputs/ED/class_'+str(normal_class)+'/' +model_name_temp)

    for f in os.listdir('./outputs/ref_vec/class_'+str(normal_class) + '/'):
      if (model_name in f) :
        os.remove(f'./outputs/ref_vec/class_'+str(normal_class) + '/{}'.format(f))
    ref_vecs.to_csv('./outputs/ref_vec/class_'+str(normal_class) + '/' +model_name_temp)


    pd.DataFrame([np.mean(inf_times), np.std(inf_times), np.mean(total_times), np.std(total_times), val_auc_min ,f1,acc]).to_csv('./outputs/inference_times/class_'+str(normal_class)+'/'+model_name_temp)

     #write out all details of model training
    cols = ['normal_class', 'ref_seed', 'weight_seed', 'num_ref_eval', 'num_ref_dist','alpha', 'lr', 'weight_decay', 'vector_size','biases', 'smart_samp', 'k', 'v', 'contam' , 'AUC', 'epoch', 'auc_min','training_time', 'f1','acc']
    params = [normal_class, args.seed, args.weight_init_seed, num_ref_eval, num_ref_dist, args.alpha, args.lr, args.weight_decay, args.vector_size, args.biases, args.smart_samp, args.k, args.v, args.contamination, val_auc, epoch+1, val_auc_min, training_time,f1,acc]
    string = './outputs/class_' + str(normal_class)
    if not os.path.exists(string):
        os.makedirs(string)
    pd.DataFrame([params], columns = cols).to_csv('./outputs/class_'+str(normal_class)+'/'+model_name)
    pd.DataFrame(train_losses).to_csv('./outputs/losses/class_'+str(normal_class)+'/'+model_name)


def init_feat_vec(model,base_ind, train_dataset, device ):
        '''
        Initialise the anchor
        Args:
            model object
            base_ind - index of training data to convert to the anchor
            train_dataset - train dataset object
            device
        '''

        model.eval()
        anchor,_,_,_ = train_dataset.__getitem__(base_ind)
        with torch.no_grad():
          anchor = model(anchor.to(device).float())

        return anchor



def create_reference(contamination, dataset_name, normal_class, task, data_path, download_data, N, seed):
    '''
    Get indexes for reference set
    Include anomalies in the reference set if contamination > 0
    Args:
        contamination - level of contamination of anomlies in reference set
        dataset name
        normal class
        task - train/test/validate
        data_path - path to data
        download data
        N - number in reference set
        seed
    '''
    indexes = []
    #train_dataset = load_dataset(dataset_name, indexes, normal_class,task, data_path, download_data) #get all training data
    
    ind = np.where(np.array(train_dataset.targets)==normal_class)[0] #get indexes in the training set that are equal to the normal class

    samp = random.sample(range(0, len(ind)), N) #randomly sample N normal data points
    final_indexes = ind[samp]

    # contamination => no anomalies so 0
    #if contamination != 0:
    #  numb = np.ceil(N*contamination)
    #  if numb == 0.0:
    #    numb=1.0

    #  con = np.where(np.array(train_dataset.targets)!=normal_class)[0] #get indexes of non-normal class
    #  samp = random.sample(range(0, len(con)), int(numb))
    #  samp2 = random.sample(range(0, len(final_indexes)), len(final_indexes) - int(numb))
    #  final_indexes = np.array(list(final_indexes[samp2]) + list(con[samp]))
    
    return final_indexes



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default='model', type=str)

    parser.add_argument('--normal_class', type=int, default = 0)
    parser.add_argument('-N', '--num_ref', type=int, default = 30)
    
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--vector_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default = 100)
    parser.add_argument('--weight_init_seed', type=int, default = 100)
    parser.add_argument('--alpha', type=float, default = 0)
    parser.add_argument('--smart_samp', type = int, choices = [0,1], default = 0)
    parser.add_argument('--k', type = int, default = 0)
    parser.add_argument('--epochs', type=int, default = 100)

    parser.add_argument('--contamination',  type=float, default=0)
    parser.add_argument('--v',  type=float, default=0.0)
    parser.add_argument('--task',  default='train', choices = ['test', 'train'])

    parser.add_argument('--eval_epoch', type=int, default=0)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--biases', type=int, default=1)

    parser.add_argument('--early_stopping', type=int, default=0)

    parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)
    
    # Modify
    parser.add_argument('--padding', type=str, 
                        default='mean', help='choose one of them : no, max, mean')
    parser.add_argument('--timespan', type=int, 
                            default = 100, help='choose of the number of timespan between data points(1000 = 1sec, 60000 = 1min)')
    parser.add_argument('--min_seq', type=int, 
                            default = 10, help='choose of the minimum number of data points in a example')
    parser.add_argument('--min_samples', type=int, default=20, 
                            help='choose of the minimum number of samples in each label')
    parser.add_argument('--dataset', default='lapras', type=str,
                            help='Dataset of choice: lapras, casas, opportunity, aras_a, aras_b')
    parser.add_argument('--aug_method', type=str, default='AddNoise', help='choose the data augmentation method')
    parser.add_argument('--aug_wise', type=str, default='Temporal', help='choose the data augmentation wise')

    parser.add_argument('--test_ratio', type=float, default=0.2, help='choose the number of test ratio')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='choose the number of test ratio')

    #parser.add_argument("--batch_size", default=64, type=int, help="Batch size per iteration")
    parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    N = args.num_ref
    num_ref_eval = N
    num_ref_dist = N

    args.model_type = 'TimeSeriesNet'

    data_type = args.dataset
    if data_type == 'lapras': 
        args.timespan = 10000
        class_num = [0, 1, 2, 3, -1]
        seq_length = 598
        channel = 7
    elif data_type == 'casas': 
        seq_length = 46
        args.aug_wise = 'Temporal2'
        class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1]
        channel = 37
    elif data_type == 'opportunity': 
        args.timespan = 1000
        class_num = [0, 1, 2, 3, 4, -1]        
        seq_length = 169
        channel = 241
    elif data_type == 'aras_a': 
        args.timespan = 1000
        seq_length = 63
        channel = 19
        class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1]


    device = torch.device(args.device)
    # loading a data
    num_classes, datalist, labellist = loading_data(data_type, args)

    store_path = 'result_files/FewSome_'+ data_type+'.xlsx'
    vis_path = 'figure/FewSome_ROC_'+data_type+'.png'
    vis_title ="ROC curves of FewSome"

    final_auroc = []
    final_aupr  = []
    final_fpr   = []
    final_de    = []  
    final_time = []

    y_onehot_test=[]
    y_score = []
    validation = []


    seed_num = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    for args.normal_class in class_num:
        auroc_a = []
        aupr_a  = []
        fpr_a   = []
        de_a    = []
        time_a = []

        testy_rs = []
        scores_rs = []

        for test_num in seed_num :

            # ##### fix random seeds for reproducibility ########
            SEED = args.seed = test_num
            np.random.seed(SEED)
            random.seed(SEED)
            #set the seed
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            #####################################################
            
            print('Data loading done.')
            print("=" * 45)
            print("Dataset:",data_type)
            print("True Class:", args.normal_class)            
            print("Seed:", args.seed)
            print("=" * 45)
            print("creating model")

            # split train / test => Train only 0 and Test  both 0 and 1
            train_list, train_label_list, test_list, test_label_list, novel_class_idx  \
                = data_generator(args, datalist, labellist)
            
            indexes = random.sample(range(0, len(train_list)), N) #randomly sample N normal data points
            

            train_dataset = Load_Dataset(train_list, train_label_list, indexes, 'train')  
            test_dataset = Load_Dataset(test_list, test_label_list, indexes, 'test')  
            
            
            #create directories
            if not os.path.exists('outputs'):
                os.makedirs('outputs')
            if not os.path.exists('outputs/models'):
                os.makedirs('outputs/models')

            string = './outputs/models/class_' + str(args.normal_class)
            if not os.path.exists(string):
                os.makedirs(string)
            if not os.path.exists('outputs/ED'):
                os.makedirs('outputs/ED')

            string = './outputs/ED/class_' + str(args.normal_class)
            if not os.path.exists(string):
                os.makedirs(string)

            if not os.path.exists('outputs/ref_vec'):
                os.makedirs('outputs/ref_vec')

            string = './outputs/ref_vec/class_' + str(args.normal_class)
            if not os.path.exists(string):
                os.makedirs(string)

            if not os.path.exists('outputs/losses'):
                os.makedirs('outputs/losses')

            string = './outputs/losses/class_' + str(args.normal_class)
            if not os.path.exists(string):
                os.makedirs(string)


            if not os.path.exists('outputs/ref_vec_by_pass/'):
                os.makedirs('outputs/ref_vec_by_pass')

            string = './outputs/ref_vec_by_pass/class_' + str(args.normal_class)
            if not os.path.exists(string):
                os.makedirs(string)


            if not os.path.exists('outputs/inference_times'):
                os.makedirs('outputs/inference_times')
            if not os.path.exists('outputs/inference_times/class_' + str(args.normal_class)):
                os.makedirs('outputs/inference_times/class_'+str(args.normal_class))



            #Initialise the model

            model = TimeSeriesNet(channel, seq_length, args.vector_size, args.biases)
            model.to(args.device)

            model_name = args.model_name + '_normal_class_' + str(args.normal_class) + '_seed_' + str(args.seed)

            #initialise the anchor
            ind = list(range(0, len(indexes)))
            #select datapoint from the reference set to use as anchor
            np.random.seed(args.epochs)
            rand_freeze = np.random.randint(len(indexes))
            base_ind = ind[rand_freeze]
            anchor = init_feat_vec(model, base_ind , train_dataset, args.device)


            criterion = ContrastiveLoss(args.alpha, anchor, args.device, args.v)

            

            print("Setting fininshed")
            print("=" * 45)
            #auc, epoch, auc_min, training_time, f1,acc, train_losses = \
            train(model,args.lr, args.weight_decay, train_dataset, test_dataset, args.epochs, criterion, args.alpha, model_name, indexes, args.normal_class, args.dataset, args.smart_samp,args.k, args.eval_epoch, args.model_type, args.batch_size, num_ref_eval, num_ref_dist, args.device)

            # Testing
            auroc_rs, aupr_rs, fpr_rs, de_re, scores, labels, infer_time = evaluate(anchor, SEED, base_ind, train_dataset, test_dataset, model, args.dataset, args.normal_class, model_name, indexes, criterion, args.alpha, num_ref_eval, device)

            print('Validation AUC is {:.3f}'.format(auroc_rs))

            auroc_a.append(auroc_rs)     
            aupr_a.append(aupr_rs)   
            fpr_a.append(fpr_rs)
            de_a.append(de_re)
            time_a.append(infer_time)

            testy_rs = testy_rs + labels
            scores_rs = scores_rs + scores

        final_auroc.append([np.mean(auroc_a), np.std(auroc_a)])
        final_aupr.append([np.mean(aupr_a), np.std(aupr_a)])
        final_fpr.append([np.mean(fpr_a), np.std(fpr_a)])
        final_de.append([np.mean(de_a), np.std(de_a)])
        final_time.append([np.mean(time_a),np.std(time_a)])

        # for visualization
        onehot_encoded = list()        
        label_binarizer = LabelBinarizer().fit(testy_rs)                    
        onehot_encoded = label_binarizer.transform(testy_rs)
        y_onehot_test.append(onehot_encoded)
        y_score.append(scores_rs)

        auroc_rs, aupr_rs, fpr_at_95_tpr_rs, detection_error_rs = calculate_acc(testy_rs, scores_rs)
        validation.append([auroc_rs,0])


    print_rs(final_auroc, final_aupr, final_fpr, final_de, final_time, store_path)


    # visualization    
    visualization_roc(class_num, y_onehot_test, y_score, vis_title, vis_path)

