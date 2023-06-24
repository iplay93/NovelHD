
from sklearn.metrics import roc_auc_score
import logging
import time
import torch
import torch.optim as optim
import numpy as np
from ood_metrics import auroc, aupr, fpr_at_95_tpr, detection_error

class AETrainer():

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda'):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device


    def train(self, train_loader, ae_net):
        # Set device for network
        ae_net = ae_net.to(self.device)


        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        print('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
        for epoch in range(self.n_epochs):

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1
            
            
            scheduler.step()
            if epoch in self.lr_milestones:
                print('LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        pretrain_time = time.time() - start_time
        print('Pretraining time: %.3f' % pretrain_time)
        print('Finished pretraining.')

        return ae_net

    def test(self, test_loader, ae_net):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Testing
        print('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1

        print('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        auc = roc_auc_score(labels, scores)
        print('Test set AUC: {:.2f}%'.format(100. * auc))
        print('Test set AUC2: {:.2f}%'.format(100. * auroc(scores, labels)))
        print('Test set AUPR: {:.2f}%'.format(100. * aupr(scores, labels)))
        print('Test set FPR: {:.2f}%'.format(100. * fpr_at_95_tpr(scores, labels)))
        print('Test set DE: {:.2f}%'.format(100. * detection_error(scores, labels)))
        
        test_time = time.time() - start_time
        print('Autoencoder testing time: %.3f' % test_time)
        print('Finished testing autoencoder.')

        return auroc(scores, labels), aupr(scores, labels), \
                fpr_at_95_tpr(scores, labels), detection_error(scores, labels)