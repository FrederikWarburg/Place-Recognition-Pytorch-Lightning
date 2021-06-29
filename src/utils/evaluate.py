import numpy as np
from sklearn.neighbors import NearestNeighbors
from torchmetrics import Metric
import torch


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap

def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.

         Usage: 
           map = compute_map (ranks, gnd) 
                 computes mean average precsion (map) only
        
           map, aps, pr, prs = compute_map (ranks, gnd, kappas) 
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
        
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def compute_map_and_print(dataset, ranks, gnd, kappas=[1, 5, 10]):
    # old evaluation protocol
    if dataset.startswith('oxford5k') or dataset.startswith('paris6k'):
        map, aps, _, _ = compute_map(ranks, gnd)
        print('>> {}: mAP {:.2f}'.format(dataset, np.around(map*100, decimals=2)))

    # new evaluation protocol
    elif dataset.startswith('roxford5k') or dataset.startswith('rparis6k'):
        
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
            gnd_t.append(g)
        mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, kappas)

        print('>> {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(dataset, kappas, np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))

"""
def recall(ranks, pidxs, qidxs, ks):
    recall_at_k = np.zeros(len(ks))
    for qidx in range(ranks.shape[1]):
        for i, k in enumerate(ks):
            if np.sum(np.in1d(ranks[:k, qidx], pidxs[qidx])) > 0:
                recall_at_k[i:] += 1
                break
    recall_at_k /= ranks.shape[0]
    return recall_at_k
"""
def recall(ranks, pidx, ks):

	recall_at_k = np.zeros(len(ks))
	for qidx in range(ranks.shape[0]):

		for i, k in enumerate(ks):
			if np.sum(np.in1d(ranks[qidx,:k], pidx[qidx])) > 0:
				recall_at_k[i:] += 1
				break

	recall_at_k /= ranks.shape[0]

	return recall_at_k

def apk(pidx, rank, k):
    if len(rank)>k:
        rank = rank[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(rank):
        if p in pidx and p not in rank[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(pidx), k)

def mapk(ranks, pidxs, k):
    return np.mean([apk(a,p,k) for a,p in zip(pidxs, ranks)])

class MeanAveragePrecision(Metric):
    def __init__(self, posDistThr = 25, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("q_embed", default=[], dist_reduce_fx='cat')
        self.add_state("db_embed", default=[], dist_reduce_fx='cat')
        self.add_state("utmQ", default=[], dist_reduce_fx='cat')
        self.add_state("utmDb", default=[], dist_reduce_fx='cat')
        self.add_state("q_index", default=[], dist_reduce_fx='cat')
        self.add_state("db_index", default=[], dist_reduce_fx='cat')
        
        self.posDistThr = posDistThr

    def update(self, embeddings: torch.Tensor, index: torch.Tensor, utm: torch.Tensor):
        
        query = index[1] == -1
        database = ~query

        # split into query and database
        q_embed = embeddings[query]
        db_embed = embeddings[database]
        utmQ = utm[query]
        utmDb = utm[database]
        q_index = index[0][query]
        db_index = index[1][database]
        
        self.q_embed.append(q_embed) 
        self.db_embed.append(db_embed)
        self.q_index.append(q_index)
        self.db_index.append(db_index)
        self.utmQ.append(utmQ)
        self.utmDb.append(utmDb)

    def compute(self):

        q_embed = self.q_embed.view(-1, self.q_embed.shape[-1]).cpu().numpy()
        db_embed = self.db_embed.view(-1, self.db_embed.shape[-1]).cpu().numpy()
        utmQ = self.utmQ.view(-1, 2).cpu().numpy()
        utmDb = self.utmDb.view(-1, 2).cpu().numpy()
        q_index = self.q_index.view(-1).cpu()
        db_index = self.db_index.view(-1).cpu()

        # remove duplicates
        q_keys = {}
        for i, val in enumerate(q_index.numpy()):
            if val not in q_keys:
                q_keys[val] = i
        
        db_keys = {}
        for i, val in enumerate(db_index.numpy()):
            if val not in db_keys:
                db_keys[val] = i

        q_index = np.array([q_keys[i] for i in q_keys.values()],dtype=int)
        db_index = np.array([db_keys[i] for i in db_keys.values()],dtype=int)

        q_embed = q_embed[q_index]
        db_embed = db_embed[db_index]
        utmQ = utmQ[q_index]
        utmDb = utmDb[db_index]

        if len(utmQ) == 0 or len(utmDb) == 0: 
            print("\n==>{} queries and {} db items found\n".format(len(utmQ), len(utmDb)))
            return np.zeros(3), np.zeros(4)

        neigh = NearestNeighbors(algorithm = 'brute')
        neigh.fit(utmDb)
        _, pidxs = neigh.radius_neighbors(utmQ, self.posDistThr)

        dist = q_embed @ db_embed.T
        ranks = np.argsort(-dist, axis=1)

        mAPs = [mapk(ranks, pidxs, k = k) for k in [5, 10, 20]]
        recalls = recall(ranks, pidxs, ks= [1, 5, 10, 20])

        return mAPs, recalls
