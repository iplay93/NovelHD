import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, \
    average_precision_score, accuracy_score, precision_score,f1_score,recall_score


from util import calculate_acc_rv

def Trainer_class(model, model_optimizer, classifier, classifier_optimizer, 
                            train_dl, device, logger, configs, experiment_log_dir):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, configs.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, model_optimizer, classifier, 
                    classifier_optimizer, criterion, train_dl, configs, device)
        if epoch % 50 == 0 : 
            logger.debug(f'Train Loss   : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n')
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'classifier_state_dict': classifier.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    logger.debug("\n################## Training is Done! #########################")
    return model

def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)



def model_train(model, model_optimizer, classifier, classifier_optimizer, criterion, 
                train_loader, configs, device):
    total_loss = []
    total_acc = []
    model.train()
    classifier.train()
    
    for batch_idx, (data, labels) in enumerate(train_loader):

        batch_size = data.shape[0]
        # send to device
        data, labels = data.float().to(device), labels.long().to(device) # data: [128, 1, 178], labels: [128]

        # optimizer
        model_optimizer.zero_grad()
        classifier_optimizer.zero_grad()


        _, s_t = model(data)                 


        loss = criterion(s_t.float(), labels)

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
    

    total_loss = torch.tensor(total_loss).mean()


    return np.mean(np.array(total_loss)), np.mean(np.array(total_acc))


def model_evaluate_class(model, classifier, test_dl, device):
    model.eval()
    classifier.eval()
    total_acc = []
    total_f1 = []
    total_auc = []
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
        for (data, labels) in test_dl:
        # send to device
            data, labels = data.float().to(device), labels.long().to(device) # data: [128, 1, 178], labels: [128]

            _, s_t = model(data)        

            acc_bs = labels.eq(s_t.detach().argmax(dim=1)).float().mean()
            onehot_label = torch.nn.functional.one_hot(labels)
            pred_numpy = s_t.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()

            try:
                auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy,
                                   average="macro", multi_class="ovr")
            except:
                auc_bs = np.float(0)

            pred_numpy = np.argmax(pred_numpy, axis=1)

            total_acc.append(acc_bs)
            total_auc.append(auc_bs)

            pred = s_t.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())
            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))
            #auroc, fpr, f1, acc = calculate_acc_rv(shift_labels.cpu().numpy().tolist(), pred.cpu().numpy().tolist())

            #total_f1.append(f1)

    labels_numpy_all = labels_numpy_all[1:]
    pred_numpy_all = pred_numpy_all[1:]


    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', )
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', )
    F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', )
    acc = accuracy_score(labels_numpy_all, pred_numpy_all, )


    total_acc = torch.tensor(total_acc).mean()
    total_auc = torch.tensor(total_auc).mean()

    performance = [acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100]
    print('MLP Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f'
          % (acc*100, precision * 100, recall * 100, F1 * 100, total_auc*100))


    return total_acc, total_auc, F1, trgs



