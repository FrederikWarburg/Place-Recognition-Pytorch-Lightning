import pytorch_lightning as pl
from utils.profiler import PassThroughProfiler
from models import configure_model, get_model_parameters
import torchvision.transforms as transforms
import torch
from pytorch_metric_learning import distances
from utils.metric_learning import TripletMarginMiner
import math
import numpy as np
from utils.loss import configure_metric_loss

class PL_LifeLongAlignment(pl.LightningModule):

    def __init__(self, args, profiler = None, metric = None):
        super().__init__()

        self.args = args
        self.model = configure_model(args)
        self.profiler = profiler or PassThroughProfiler()

        ### pytorch-metric-learning stuff ###
        if args.distance == 'dot':
            distance = distances.DotProductSimilarity()
        elif args.distance == 'euclidean':
            distance = distances.LpDistance(p=2, normalize_embeddings = True, power=2)
        
        self.global_retrieval_criterion = configure_metric_loss(args.loss, args.distance, args.margin)
        self.global_metric = metric
        
        self.miner = TripletMarginMiner(margin=args.margin, 
                                collect_stats=True, 
                                type_of_triplets=args.miner, 
                                posDistThr=self.args.posDistThr, 
                                negDistThr=self.args.negDistThr,
                                distance=distance) 

        self.counter = 0
        self.val_counter = 0
        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
            
    def forward(self, x):

        output = self.model(x)

        return output

    def on_train_start(self):
        self.logger.log_hyperparams(vars(self.args), {"val_map/map@5": 0, "test_map/map@5": 0})
            
    def on_validation_epoch_start(self):

        self.global_metric.reset()

    def on_test_epoch_start(self):

        self.global_metric.reset()

    def training_step(self, batch, batch_idx):

        x, y = batch
        n = len(x)
        b,c,h,w = x[0].shape
        x = torch.stack(x).view(b*n, c, h, w)
        y = torch.stack(y).view(b*n, 2)
        
        output = self(x)

        indices_tuple = self.get_indices_tuple(output["global_embed"], y)

        loss = self.compute_loss(output, y, indices_tuple)

        # add images to tensorboard every epoch
        if self.current_epoch == self.counter:
            self.log_triplets(x, indices_tuple)   
            self.counter += 1

        return loss

    def training_epoch_end(self, outputs):
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):

        x, index, utm = batch

        output = self(x)

        self.global_metric.update(output["global_embed"], index, utm)

        return {}

    def validation_epoch_end(self, outputs):
   
        mAPs, recalls = self.global_metric.compute()  
        for i, k in enumerate([5, 10, 20]):
            self.log("val_map/map@{}".format(k), mAPs[i], prog_bar=True)

        for i, k in enumerate([1, 5, 10, 20]):
            self.log("val_recall/recall@{}".format(k), recalls[i])

    def test_step(self, batch, batch_idx):

        x, index, utm = batch

        output = self(x)

        self.global_metric.update(output["global_embed"], index, utm)

        return {}

    def test_epoch_end(self, outputs):
        
        mAPs, recalls = self.global_metric.compute()  
        for i, k in enumerate([5, 10, 20]):
            self.log("test_map/map@{}".format(k), mAPs[i], prog_bar=True)

        for i, k in enumerate([1, 5, 10, 20]):
            self.log("test_recall/recall@{}".format(k), recalls[i])

    def get_indices_tuple(self, y_hat, y):

        indices_tuple = self.miner(embeddings = y_hat, labels = y)

        self.log("tuple_stats/an_dist", self.miner.neg_pair_dist)
        self.log("tuple_stats/ap_dist", self.miner.pos_pair_dist)
        self.log("tuple_stats/n_triplets", self.miner.num_triplets)

        return indices_tuple

    def compute_loss(self, output, y, indices_tuple):
            
        global_retrieval_loss = self.global_retrieval_criterion(output["global_embed"], y, indices_tuple)
        self.log('train_global_retrieval_loss', global_retrieval_loss, on_step=True, on_epoch=True)

        return global_retrieval_loss

    def configure_optimizers(self):

        parameters = get_model_parameters(self.model, self.args)

        # define optimizer
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(parameters, self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(parameters, self.args.lr, weight_decay=self.args.weight_decay)

        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(-0.01)),
            'name': 'exponential_lr'
        }

        return [optimizer], [lr_scheduler]

    def log_triplets(self, x, indices_tuple):

        tensorboard  = self.logger.experiment

        # display triplets on tensorboard
        len_ = len(indices_tuple[0])
        index = np.random.choice(len_, min(5,len_), replace=False)
        for i, idx in enumerate(index):

            a = self.inv_normalize(x[indices_tuple[0][idx]])
            p = self.inv_normalize(x[indices_tuple[1][idx]])
            n = self.inv_normalize(x[indices_tuple[2][idx]])

            tensorboard.add_images("triplets/{}".format(i), torch.stack([a,p,n]), self.current_epoch)