from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np
from ood_metrics import auroc, aupr, fpr_at_95_tpr, detection_error

class DeepSVDDTrainer():

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__()
        self. optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
    
    def calc_metrics(testy, scores):
        return auroc(scores, testy), aupr(scores, testy), fpr_at_95_tpr(scores, testy), detection_error(scores, testy)


    def train(self, train_loader, net):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            print('Center c initialized.')

        # Training
        print('Starting training...')
        start_time = time.time()
        net.train()
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
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            scheduler.step()
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))


            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            #print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
#                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time
        print('Training time: %.3f' % self.train_time)

        print('Finished training.')

        return net

    def test(self, test_loader , net):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Testing
        print('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        print('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels).tolist()
        scores = np.array(scores).tolist()

        self.test_auc = roc_auc_score(labels, scores)
        print('Test set AUC: {:.2f}'.format(self.test_auc))
        print('Test set AUC2: {:.2f}'.format(auroc(scores, labels)))
        print('Test set AUPR: {:.2f}'.format(aupr(scores, labels)))
        print('Test set FPR: {:.2f}'.format(fpr_at_95_tpr(scores, labels)))
        print('Test set DE: {:.2f}'.format(detection_error(scores, labels)))
        print('Finished testing.')

        return auroc(scores, labels), aupr(scores, labels), \
                fpr_at_95_tpr(scores, labels), detection_error(scores, labels), scores, labels

    def init_center_c(self, train_loader: DataLoader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)