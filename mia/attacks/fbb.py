import numpy as np
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

from mia.attacks.utils import *
from mia.attacks.base import BaseAttacker


class FullBlackBox():
    def __init__(self, syn, batch_size = 2000):
        super(FullBlackBox, self).__init__()

        self.K = 2000
        self.batch_size = batch_size
        self.nn_obj = NearestNeighbors(n_neighbors=self.K, n_jobs=-1)
        self.nn_obj.fit(syn)

    def _find_knn(self, query_data):
        '''
        :param nn_obj: Nearest Neighbor object
        :param query_data: synthesized data
        :return:
            dist: distance between query samples to its KNNs among generated samples
            idx: index of the KNNs
        '''
        dist = []
        idx = []
        plus = 1 if len(query_data) % self.batch_size != 0  else 0
        for i in range(len(query_data) // self.batch_size + plus):
            x_batch = query_data[i * self.batch_size:(i + 1) * self.batch_size]
            if x_batch.shape[0] < self.batch_size:
                x_batch = np.reshape(x_batch, [x_batch.shape[0], -1])
            else:
                x_batch = np.reshape(x_batch, [self.batch_size, -1])
            dist_batch, idx_batch = self.nn_obj.kneighbors(x_batch)
            dist.append(dist_batch)
            idx.append(idx_batch)

        try:
            dist = np.concatenate(dist)
            idx = np.concatenate(idx)
        except:
            dist = np.array(dist)
            idx = np.array(idx)
        return dist, idx


class FullBlackBoxAttacker(BaseAttacker):
    def attack(self, train, test, metadata, K):
        pos_query, neg_query = self.query(train, test)

        syn = self.synthesizer.sample(K)
        pos_loss = compute_distance(syn, pos_query, metadata)
        neg_loss = compute_distance(syn, neg_query,  metadata)
        labels = np.concatenate((np.zeros((len(neg_loss),)), np.ones((len(pos_loss),))))
        results = np.concatenate((-neg_loss, -pos_loss))

        auc = metrics.roc_auc_score(labels, results)
        ap = metrics.average_precision_score(labels, results)

        return auc, ap