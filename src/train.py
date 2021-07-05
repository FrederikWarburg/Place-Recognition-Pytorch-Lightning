
import pytorch_lightning as pl
import argparse
import pprint
from loguru import logger as loguru_logger
import torchvision.models as models
import torch
import os
from datetime import datetime

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from lightning.data import PlaceRecognitionDataModule
from lightning.pl_lifelongalignment import PL_LifeLongAlignment
from utils.profiler import build_profiler
from utils.evaluate import MeanAveragePrecision

training_dataset_names = ['msls', 'aerialstreet']
test_datasets_names = ['msls', 'aerialstreet']

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names.extend(['vit_small', 'vit_base'])
pool_names = ['mac', 'spoc', 'gem', 'gemmp']
local_pool_names = ['ws', 'wgem']
loss_names = ['contrastive', 'triplet']
optimizer_names = ['sgd', 'adam']

def parse_args():

    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

    # export directory, training and val datasets, test datasets
    parser.add_argument('--directory', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')
    parser.add_argument('--data_dir', metavar='data_dir', default="/home/frwa/Desktop/data/MSLS",
                        help='destination to data')
    parser.add_argument('--training-dataset', '-d', metavar='DATASET', default='msls', choices=training_dataset_names,
                        help='training dataset: ' + 
                            ' | '.join(training_dataset_names) +
                            ' (default: msls)')
    parser.add_argument('--test-datasets', '-td', metavar='DATASETS', default='msls', choices=test_datasets_names,
                        help='comma separated list of test datasets: ' + 
                            ' | '.join(test_datasets_names) + 
                            ' (default: msls)')
    parser.add_argument('--test-freq', default=20, type=int, metavar='N', 
                        help='run test evaluation every N epochs (default: 1)')

    parser.add_argument('--train_cities', metavar='TRAIN_CITIES', default='', help='city mode') #"zurich,london,boston"
    parser.add_argument('--val_cities', metavar='VAL_CITIES', default='', help='city mode')
    parser.add_argument('--test_cities', metavar='TEST_CITIES', default='', help='city mode')
    parser.add_argument('--posDistThr', metavar='POSITIVEDIST', default=15, help='tuple mining', type=int)
    parser.add_argument('--negDistThr', metavar='NEGATIVEDIST', default=25, help='tuple mining', type=int)

    # network architecture and initialization options
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')
    parser.add_argument('--pool', '-p', metavar='POOL', default='gem', choices=pool_names,
                        help='pooling options: ' +
                            ' | '.join(pool_names) +
                            ' (default: gem)')
    parser.add_argument('--regional', '-r', dest='regional', action='store_true',
                        help='train model with regional pooling using fixed grid')
    parser.add_argument('--whitening', '-w', dest='whitening', action='store_true',
                        help='train model with learnable whitening (linear layer) after the pooling')
    parser.add_argument('--not-pretrained', dest='pretrained', action='store_false',
                        help='initialize model with random weights (default: pretrained on imagenet)')
    parser.add_argument('--loss', '-l', metavar='LOSS', default='triplet',
                        choices=loss_names,
                        help='training loss options: ' +
                            ' | '.join(loss_names) +
                            ' (default: triplet)')
    parser.add_argument('--margin', '-lm', metavar='LM', default=0.1, type=float,
                        help='loss margin: (default: 0.1)')

    # train/val options specific for image retrieval learning
    parser.add_argument('--image-size', default=640, type=int, metavar='N',
                        help='maximum size of longer image side used for training (default: 320)')

    # standard train/val options
    parser.add_argument('--workers', '-j', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--batch_size', '-b', default=8, type=int, metavar='N', 
                        help='number of (q,p,n1,...,nN) tuples in a mini-batch (default: 5)')
    parser.add_argument('--update_every', '-u', default=5, type=int, metavar='N',
                        help='update model weights every N batches, used to handle really large batches, ' + 
                            'batch_size effectively becomes update_every x batch_size (default: 1)')
    parser.add_argument('--optimizer', '-o', metavar='OPTIMIZER', default='adam',
                        choices=optimizer_names,
                        help='optimizer options: ' +
                            ' | '.join(optimizer_names) +
                            ' (default: adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                        metavar='LR', help='initial learning rate (default: 1e-6)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float,
                        metavar='W', help='weight decay (default: 1e-6)')
    parser.add_argument('--resume', default='', type=str, metavar='FILENAME',
                        help='name of the latest checkpoint (default: None)')
    parser.add_argument('--seed', default=42, type=int,
                        metavar='seed', help='seed')
    parser.add_argument('--miner', default='none', type=str,
                        metavar='miner', help='which miner to use (default: all)')
    parser.add_argument('--distance', dest='distance', type = str, default='euclidean',
                        help='distance metric during training')
    
    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    pprint.pprint(vars(args))

    pl.seed_everything(args.seed)  # reproducibility

    loguru_logger.info(f"Args and config initialized!")
    
    # lightning module
    profiler = 'simple'
    metric = MeanAveragePrecision(args.negDistThr)
    
    if os.path.isfile(args.resume):
        print("Load from checkount {}".format(args.resume))
        model = PL_LifeLongAlignment.load_from_checkpoint(checkpoint_path=args.resume, args=args, profiler=profiler, metric = metric, strict = True)
    else:
        model = PL_LifeLongAlignment(args, profiler=profiler, metric = metric)

    loguru_logger.info(f"LifeLongAlignment-lightning initialized!")

    # lightning data
    data_module = PlaceRecognitionDataModule(**vars(args))
    loguru_logger.info(f"DataModule initialized!")

    logger = pl.loggers.TensorBoardLogger('lightning_logs/', 
                                            name=args.arch, 
                                            version=datetime.now().strftime("%d-%m_%H-%M-%S"),
                                            default_hp_metric=False)

    # lightning trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val_map/map@5',
        dirpath='{}_checkpoints'.format(logger.log_dir),
        filename=args.arch + '-{epoch:02d}-{val_map@5:.2f}',
        save_top_k=1,
        mode='max',
        save_last=True
    )

    # scale learning rate 
    args.lr = args.lr * args.batch_size * args.update_every * torch.cuda.device_count()

    callbacks = [LearningRateMonitor(logging_interval='step'), checkpoint_callback]
    
    # freeze model paramters
    trainer = pl.Trainer.from_argparse_args(args, 
                                accelerator='ddp',
                                accumulate_grad_batches=args.update_every,
                                precision=16,
                                max_epochs=args.epochs,
                                gpus=torch.cuda.device_count(),
                                check_val_every_n_epoch=1,
                                progress_bar_refresh_rate=20,
                                logger=logger, 
                                #plugins=DDPPlugin(find_unused_parameters=False),
                                callbacks=callbacks,
                                profiler=profiler
                                )

    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)

    loguru_logger.info(f"Start testing!")
    trainer.test(model, datamodule=data_module)

