import pandas as pd
from os.path import join
import numpy as np
import torch.utils.data as data
import sys
from sklearn.neighbors import NearestNeighbors

from datasets.datahelpers import default_loader, imresize

class BaseDataset(data.Dataset):
 

    def __init__(self, name, mode='train', imsize=None, transform=None, loader=default_loader, posDistThr=10, negDistThr=25, root_dir = 'data'):
                        
        self.qidxs = [] 
        self.pidxs = []  
        self.clusters = [] 

        # hyper-parameters
        self.posDistThr = posDistThr
        self.negDistThr = negDistThr
        self.imsize = imsize

        self.transform = transform
        self.loader = loader

        # flags
        self.name = name
        self.mode = 'test' if mode in ('test', 'val') else 'train'

        # other
        self.transform = transform

        # load query / database data
        #TODO: rename on disk
        qData = pd.read_csv(join(root_dir, self.mode, 'query', 'data.csv'), index_col=0)
        
        # remove offset
        qData['easting'] = qData['X1'] - 690000
        qData['northing'] = qData['Y1'] - 6170000
        qData['altitude'] = qData['Z1'] - 0
        qData['key'] = qData['Image filename']

        #TODO: include filenames
        dbData = pd.read_csv(join(root_dir, self.mode, 'database', 'data.csv'), index_col=0)
        
        # append image keys with full path
        self.qImages = np.asarray([join(root_dir, self.mode, 'query', 'images', key) for key in qData['key'].values])
        self.dbImages = np.asarray([join(root_dir, self.mode, 'database', 'images', key) for key in dbData['key'].values])

        # utm coordinates
        self.utmQ = qData[['easting', 'northing']].values.reshape(-1,2)
        self.utmDb = dbData[['easting', 'northing']].values.reshape(-1,2)

        # find positive images for training
        neigh = NearestNeighbors(algorithm = 'brute')
        neigh.fit(self.utmDb)
        _, pI = neigh.radius_neighbors(self.utmQ, self.posDistThr)

        if self.mode == 'train':
            _, nI = neigh.radius_neighbors(self.utmQ, self.negDistThr)

        for qidx in range(len(qData)):
            
            # the query image has at least one positive
            if len(pI[qidx]) > 0:
                
                self.qidxs.append(qidx)

                #TODO: include cas threshold
                self.pidxs.append([p for p in pI[qidx]])
                
                # in training we have two thresholds, one for finding positives and one for finding images that we are certain are negatives.
                if self.mode == 'train':
                    self.clusters.append([n for n in nI[qidx]])
            
        # cast to np.arrays for indexing during training
        self.qidxs = np.asarray(self.qidxs)
        self.pidxs = np.asarray(self.pidxs, dtype=object)
        self.clusters = np.asarray(self.clusters, dtype=object)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        
        raise NotImplementedError

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name and mode: {} {}\n'.format(self.name, self.mode)
        fmt_str += '    Number of query images: {}\n'.format(len(self.qImages))
        fmt_str += '    Number of database images: {}\n'.format(len(self.dbImages))
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class TrainDataset(BaseDataset):

    def __init__(self, name, mode, imsize=None, transform=None, loader=default_loader, posDistThr=10, negDistThr=25, root_dir = 'data'):
        super().__init__(name, mode, imsize, transform, loader, posDistThr, negDistThr, root_dir)

    def __len__(self):

        return len(self.qidxs)

    def __getitem__(self, index):

        qidx =  self.qidxs[index]
        pidx =  self.pidxs[index]
        
        pidx =  np.random.choice(pidx, 1)[0]
        
        qpath, utmQ = self.qImages[qidx], self.utmQ[qidx]
        ppath, utmDb = self.dbImages[pidx], self.utmDb[pidx]

        output = []
        output.append(self.loader(qpath))
        output.append(self.loader(ppath))

        target = []
        target.append(utmQ)
        target.append(utmDb)

        if self.imsize is not None:
            output = [imresize(img, self.imsize) for img in output]
        
        if self.transform is not None:
            output = [self.transform(output[i]) for i in range(len(output))]
        
        return output, target

class TestDataset(BaseDataset):

    def __init__(self, name, mode, imsize=None, transform=None, loader=default_loader, posDistThr=10, negDistThr=25, root_dir = 'data'):
        super().__init__(name, mode, imsize, transform, loader, posDistThr, negDistThr, root_dir)

    def __len__(self):
        # the dataset is the queries followed by the database images
        return len(self.qidxs) + len(self.dbImages)

    def __getitem__(self, index):
        
        if index < len(self.qidxs):
            path = self.qImages[self.qidxs[index]]
            utm = self.utmQ[self.qidxs[index]]            
            index = [index, -1]
        else:
            path = self.dbImages[index - len(self.qidxs)]
            utm = self.utmDb[index - len(self.qidxs)]
            index = [-1, index - len(self.qidxs)]

        img = self.loader(path)

        if self.imsize is not None:
            img = imresize(img, self.imsize)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, index, utm

        
