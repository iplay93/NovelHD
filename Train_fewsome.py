import torch
from models.FewSome_Net import *
import os
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
import torch.optim as optim
from Test_fewsome import evaluate
import random
import time
from data_preprocessing.dataloader import loading_data
from torch.utils.data import DataLoader, Dataset
import math, random
from sklearn.model_selection import train_test_split
from data_preprocessing.dataloader import count_label_labellist

test_ratio = 0.2

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_list, label_list):
        super(Load_Dataset, self).__init__()
        
        X_train = data_list
        y_train = label_list

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

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

    def __getitem__(self, index: int, seed = 1, base_ind=-1, indexes=[]):
        
        base=False
        data1, target = self.x_data[index], int(self.y_data[index])
        data1 = torch.FloatTensor(data1)

        

        if len(indexes)  > 0 : # when training
            np.random.seed(seed)
            ind = np.random.randint(len(indexes))
            c=1
            while (ind == index):
                np.random.seed(seed * c)
                ind = np.random.randint(len(indexes))
                c=c+1

            if ind == base_ind:
              base = True

            data2 , target2 = self.x_data[ind], int(self.y_data[ind])
            data2 = torch.FloatTensor(data2)
            label = torch.FloatTensor([0])
            
        else:
            data2 = torch.Tensor([1])
            label = target

        return data1, data2, label, base
        

    def __len__(self):
        return self.len
    
def data_generator(args, configs, datalist, labellist):
    test_ratio = args.test_ratio
    seed =  args.seed 

    # Split train and valid dataset
    train_list, test_list, train_label_list, test_label_list = train_test_split(datalist, 
                                                                                labellist, test_size=test_ratio, stratify= labellist, random_state=seed) 


    print(f"Train Data: {len(train_list)} --------------")
    exist_labels, _ = count_label_labellist(train_label_list)

    print(f"Test Data: {len(test_list)} --------------")
    count_label_labellist(test_label_list) 
    
    train_list = torch.tensor(train_list).cuda().cpu()
    train_label_list = torch.tensor(train_label_list).cuda().cpu()

    test_list = torch.tensor(test_list).cuda().cpu()
    test_label_list = torch.tensor(test_label_list).cuda().cpu()

 
    if(args.one_class_idx != -1): # one-class
        sup_class_idx = [x for x in exist_labels]
        known_class_idx = [args.one_class_idx]
        novel_class_idx = [item for item in sup_class_idx if item not in set(known_class_idx)]
        
        train_list = train_list[np.where(train_label_list == args.one_class_idx)]
        train_label_list = train_label_list[np.where(train_label_list == args.one_class_idx)]

        valid_list = test_list[np.where(test_label_list == args.one_class_idx)]
        valid_label_list = test_label_list[np.where(test_label_list == args.one_class_idx)]

        # only use for testing novelty
        test_list = test_list[np.where(test_label_list != args.one_class_idx)]
        test_label_list = test_label_list[np.where(test_label_list != args.one_class_idx)]

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

        
    # train_label_list[:] = 0
    # valid_label_list[:] = 0
    # test_label_list[:] = 1

    # build data loader (N, T, C) -> (N, C, T)
   
    train_dataset = Load_Dataset(train_list, train_label_list)    
    valid_dataset = Load_Dataset(valid_list, valid_label_list)    
    test_dataset = Load_Dataset(test_list, test_label_list)    

    

    return train_dataset, valid_dataset, test_dataset, novel_class_idx  



class ContrastiveLoss(torch.nn.Module):
    def __init__(self, v=0.0,margin=0.8):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.v = v

    def forward(self, output1, vectors, feat1, label, alpha):
        euclidean_distance = torch.FloatTensor([0]).cuda()

        #get the euclidean distance between output1 and all other vectors
        for i in vectors:
          euclidean_distance += (F.pairwise_distance(output1, i)/ torch.sqrt(torch.Tensor([output1.size()[1]])).cuda())


        euclidean_distance += alpha*((F.pairwise_distance(output1, feat1)) /torch.sqrt(torch.Tensor([output1.size()[1]])).cuda() )

        #calculate the margin
        marg = (len(vectors) + alpha) * self.margin


        #if v > 0.0, implement soft-boundary
        if self.v > 0.0:
            euclidean_distance = (1/self.v) * euclidean_distance
        #calculate the loss

        loss_contrastive = ((1-label) * torch.pow(euclidean_distance, 2) * 0.5) + ( (label) * torch.pow(torch.max(torch.Tensor([ torch.tensor(0), marg - euclidean_distance])), 2) * 0.5)

        return loss_contrastive

def create_batches(lst, n):

    # looping till length l
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def init_feat_vec(model,base_ind, train_dataset ):

        model.eval()
        feat1,_,_,_ = train_dataset.__getitem__(base_ind)

        with torch.no_grad():
          feat1 = model(feat1.cuda().float()).cuda()

        return feat1


def train(model,lr, weight_decay, train_dataset, valid_dataset, epochs, criterion, alpha, 
        indexes, normal_class, dataset_name, smart_samp,k, 
        eval_epoch, bs, num_ref_eval, num_ref_dist):
    
    device='cuda'
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
    train_losses = []


    ind = list(range(0, len(indexes)))
    #select datapoint from the reference set to use as anchor
    np.random.seed(epochs)
    rand_freeze = np.random.randint(len(indexes))
    base_ind = ind[rand_freeze]
    feat1 = init_feat_vec(model,base_ind , train_dataset)

    patience = 0
    max_patience = 2
    best_val_auc = 0
    best_val_auc_min = 0
    best_f1=0
    best_acc=0
    max_iter = 0
    patience2 = 10
    stop_training = False

    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        loss_sum = 0
        print("Starting epoch " + str(epoch+1))
        np.random.seed(epoch)

        np.random.shuffle(ind)
        

        batches = list(create_batches(ind, bs))

        for i in range(int(np.ceil(len(ind) / bs))):

            model.train()

            for batch_ind,index in enumerate(batches[i]):
                
                seed = (epoch+1) * (i+1) * (batch_ind+1)
                
                data1, data2, labels, base = train_dataset.__getitem__(index, seed, base_ind, indexes)

                # Forward
                data1 = data1.to(device)
                data2 = data2.to(device)
                labels = labels.to(device)


                if (index == base_ind):
                  output1 = feat1
                else:
                  output1 = model(data1.cuda().float())

                if (smart_samp == 0) & (k > 1):

                  vecs=[]

                  ind2 = ind.copy()
                  np.random.seed(seed)
                  np.random.shuffle(ind2)

                  for j in range(0, k):
                    if (ind2[j] != base_ind) & (index != ind2[j]):
                      output2=model(train_dataset.__getitem__(ind2[j], seed, base_ind, indexes)[0].to(device).float())
                      vecs.append(output2)

                elif smart_samp == 0:

                  if (base == True):
                    output2 = feat1
                  else:
                    output2 = model(data2.cuda().float())

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
                      output2=model(train_dataset.__getitem__(ind2[j], seed, base_ind, indexes)[0].to(device).float())
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
                    loss = criterion(output1,vecs,feat1,labels,alpha)
                else:
                    loss = loss + criterion(output1,vecs,feat1,labels,alpha)

            loss_sum+= loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


        train_losses.append((loss_sum / len(ind)))

        print("Epoch: {}, Train loss: {:.3f}".format(epoch+1, train_losses[-1]))


        if (eval_epoch == 1):
            val_auc, val_loss, val_auc_min, f1, acc,df, ref_vecs = evaluate(feat1, seed, base_ind, train_dataset, valid_dataset, 
                                                                            model, dataset_name, normal_class,  indexes, criterion, alpha, num_ref_eval)
            print('Validation AUC is {}'.format(val_auc))
            print("Epoch: {}, Validation loss: {}".format(epoch+1, val_loss))
            if val_auc_min > best_val_auc_min:
                best_val_auc = val_auc
                best_val_auc_min = val_auc_min
                best_epoch = epoch
                best_f1 = f1
                best_acc = acc
                best_df=df
                max_iter = 0

                training_time = time.time() - start_time
            

            else:
                max_iter+=1

            if max_iter == patience2:
                break

        elif args.early_stopping ==1:
            if epoch > 1:
              decrease = (((train_losses[-3] - train_losses[-2]) / train_losses[-3]) * 100) - (((train_losses[-2] - train_losses[-1]) / train_losses[-2]) * 100)

              if decrease <= 0.5:
                patience += 1


              if (patience==max_patience) | (epoch == epochs-1):
                  stop_training = True


        elif (epoch == (epochs -1)) & (eval_epoch == 0):
            stop_training = True


        if stop_training == True:
            print("--- %s seconds ---" % (time.time() - start_time))
            training_time = time.time() - start_time

            val_auc, val_loss, val_auc_min, f1,acc, df, ref_vecs = evaluate(feat1,seed, base_ind, train_dataset, valid_dataset, model, dataset_name, normal_class, indexes, criterion, alpha, num_ref_eval)

            break


    print("Finished Training")
    if eval_epoch == 1:
        print("AUC was {} on epoch {}".format(best_val_auc_min, best_epoch+1))
        return best_val_auc, best_epoch, best_val_auc_min, best_f1, best_acc,train_losses
    else:
        print("AUC was {} on epoch {}".format(val_auc_min, epoch+1))
        return val_auc, epoch, val_auc_min, training_time, f1,acc, train_losses



def write_results(model_name, normal_class, model, df, ref_vecs,num_ref_eval, num_ref_dist, val_auc, epoch, val_auc_min, training_time,f1,acc, train_losses):
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


     #write out all details of model training
    cols = ['normal_class', 'ref_seed', 'weight_seed', 'num_ref_eval', 'num_ref_dist','alpha', 'lr', 'weight_decay', 'vector_size','biases', 'smart_samp', 'k', 'v', 'contam' , 'AUC', 'epoch', 'auc_min','training_time', 'f1','acc']
    params = [normal_class, args.seed, args.weight_init_seed, num_ref_eval, num_ref_dist, args.alpha, args.lr, args.weight_decay, args.vector_size, args.biases, args.smart_samp, args.k, args.v, args.contamination, val_auc, epoch+1, val_auc_min, training_time,f1,acc]
    string = './outputs/class_' + str(normal_class)
    if not os.path.exists(string):
        os.makedirs(string)
    pd.DataFrame([params], columns = cols).to_csv('./outputs/class_'+str(normal_class)+'/'+model_name)
    pd.DataFrame(train_losses).to_csv('./outputs/losses/class_'+str(normal_class)+'/'+model_name)


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('-N', '--num_ref', type=int, default = 30)
    parser.add_argument('--num_ref_eval', type=int, default = None)
    parser.add_argument('--lr', type=float , default=1e-6)
    parser.add_argument('--vector_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--weight_init_seed', type=int, default = 1001)
    parser.add_argument('--alpha', type=float, default = 0)
    parser.add_argument('--smart_samp', type = int, choices = [0,1], default = 0)
    parser.add_argument('--k', type = int, default = 0)
    parser.add_argument('--epochs', type=int, default = 300)

    parser.add_argument('--v',  type=float, default=0.0)
    parser.add_argument('--task',  default='train', choices = ['test', 'train'])

    parser.add_argument('--eval_epoch', type=int, default=0)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--biases', type=int, default=1)
    parser.add_argument('--num_ref_dist', type=int, default=30)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)
    
    parser.add_argument('--dataset', default='lapras', type=str)
    parser.add_argument('--padding', type=str, 
                    default='mean', help='choose one of them : no, max, mean')
    parser.add_argument('--timespan', type=int, default=10000, 
                        help='choose of the number of timespan between data points (1000 = 1sec, 60000 = 1min)')
    parser.add_argument('--min_seq', type=int, 
                    default=10, help='choose of the minimum number of data points in a example')
    parser.add_argument('--min_samples', type=int, default=20, 
                    help='choose of the minimum number of samples in each label')
    parser.add_argument('--one_class_idx', type=int, default=0, 
                    help='choose of one class label number that wants to deal with. -1 is for multi-classification')
    parser.add_argument('--aug_method', type=str, default='AddNoise', 
                        help='choose the data augmentation method')
    parser.add_argument('--aug_wise', type=str, default='Temporal', 
                        help='choose the data augmentation wise')
    parser.add_argument('--test_ratio', type=float, default=0.2, 
                        help='choose the number of test ratio')
    parser.add_argument('--seed', default = 42, type=int,
                    help='seed value')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()

    dataset_name = args.dataset
    normal_class = args.one_class_idx
    N = args.num_ref

    epochs = args.epochs

    indexes = args.index
    alpha = args.alpha
    lr = args.lr
    vector_size = args.vector_size
    weight_decay = args.weight_decay
    smart_samp = args.smart_samp
    k = args.k
    weight_init_seed = args.weight_init_seed
    v = args.v
    task = args.task
    eval_epoch = args.eval_epoch
    bs = args.batch_size
    biases = args.biases
    num_ref_eval = args.num_ref_eval
    num_ref_dist = args.num_ref_dist

    # Set seed
    # ##### fix random seeds for reproducibility ########
    SEED = args.seed = 40
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    #set the seed
    torch.manual_seed(weight_init_seed)
    torch.cuda.manual_seed(weight_init_seed)
    torch.cuda.manual_seed_all(weight_init_seed)

    #####################################################

    if num_ref_eval == None:
        num_ref_eval = N
    if num_ref_dist == None:
        num_ref_dist = N
    
    if dataset_name == 'lapras': 
        args.timespan = 10000
        seq_length = 598
        channel = 7
    elif dataset_name == 'casas': 
        args.aug_wise = 'Temporal2'
        seq_length = 46
        channel = 37
    elif dataset_name == 'opportunity': 
        args.timespan = 1000
        seq_length = 169
        channel = 241
    elif dataset_name == 'aras_a': 
        args.timespan = 1000
        seq_length = 63
        channel = 19

    exec(f'from config_files.{dataset_name}_Configs import Config as Configs')
    configs = Configs()
    
    # loading a certain dataset
    num_classes, datalist, labellist = loading_data(dataset_name, args)



    #if indexes for reference set aren't provided, create the reference set.
    # if dataset_name != 'mvtec':
    #     if indexes != []:
    #         indexes = [int(item) for item in indexes.split(', ')]
    #     else:
    
    #indexes = create_reference(args, contamination, dataset_name, normal_class, 'train', N, SEED)
    
    # create train and test set
    train_dataset, valid_dataset, test_dataset, novel_class_idx = data_generator(args, configs, datalist, labellist)
    indexes = random.sample(range(0, len(train_dataset)), N) #randomly sample N normal data points


#     train_dataset = Load_Dataset(args, dataset_name, indexes, normal_class, 'train', SEED, N=N)
#     indexes = train_dataset.indexes

    
#  #test
#     val_dataset = Load_Dataset(args, dataset_name, indexes, normal_class, 'train', SEED, N=N)    

    
    #Initialise the model
    model =  TimeSeriesNet(configs, vector_size)

    # model_name = model_name + '_normal_class_' + str(normal_class) + '_seed_' + str(SEED)
    criterion = ContrastiveLoss(v)
    auc, epoch, auc_min, training_time, f1,acc, train_losses= train(model,lr, 
                 weight_decay, train_dataset, valid_dataset, epochs, criterion, alpha, 
                 indexes, normal_class, dataset_name, smart_samp,k, 
                 eval_epoch, bs, num_ref_eval, num_ref_dist)