from .diversifier_base import DiversifierBase
from sklearn.cluster import KMeans as km
from sklearn.cluster import AgglomerativeClustering as ag
import numpy as np
import logging


class Cluster(DiversifierBase):
    """ Self-supervised diversification by clustering.
        The data clusters are computed and samples with
        the most utility are drawn from each cluster
        (the cluster priority is detemined by "samplings")
        See https://arxiv.org/abs/2107.14263 for details.
    """
    
    def __init__(self,
                num_clusters,
                cluster_metric,
                merge_criterion,
                sampling_method,
                **kwargs):
        super().__init__()
        self.num_clusters = num_clusters
        clusterings = {
                'random': self.random_c,
                'kmeans': self.kmeans_c,
                'aggward': self.ag_ward_c,
                'aggmin': self.ag_min_c,
                'aggmax': self.ag_max_c,
                'aggavg': self.ag_avg_c}
        samplings = {
                'uniform': self.uniform_s,
                'proportional': self.proportional_s}
        self.clustering = clusterings[merge_criterion]
        self.metric = cluster_metric
        self.sampling = samplings[sampling_method]
        self.logger = logging.getLogger(__name__)

    def random_c(self, data_X):
        cluster_size = len(data_X)//self.num_clusters
        cluster_ids = [i*np.ones((cluster_size)) for i in range(self.num_clusters-1)]
        last_cluster_size = len(data_X)-cluster_size*(self.num_clusters-1)
        last_cluster = (self.num_clusters-1)*np.ones((last_cluster_size))
        cluster_ids.append(last_cluster)
        cluster_ids = np.concatenate(cluster_ids)
        cluster_ids = np.random.choice(
                cluster_ids,
                size=len(cluster_ids),
                replace=False)
        return cluster_ids.astype(np.int8)

    def kmeans_c(self, data_X):
        if self.metric=='cosine':
            norm_X = np.linalg.norm(data_X, axis=1)[:,None]
            data_X = data_X / norm_X
        clusterer = km(
                n_clusters=self.num_clusters,
                init='random',
                algorithm='lloyd',
                n_init=2)
        cluster_ids = clusterer.fit_predict(data_X)
        return cluster_ids

    def ag_ward_c(self, data_X):
        clusterer = ag(
                n_clusters=self.num_clusters,
                metric=self.metric,
                linkage='ward')
        cluster_ids = clusterer.fit_predict(data_X)
        return cluster_ids

    def ag_min_c(self, data_X):
        clusterer = ag(
                n_clusters=self.num_clusters,
                metric=self.metric,
                linkage='single')
        cluster_ids = clusterer.fit_predict(data_X)
        return cluster_ids

    def ag_avg_c(self, data_X):
        clusterer = ag(
                n_clusters=self.num_clusters,
                metric=self.metric,
                linkage='average')
        cluster_ids = clusterer.fit_predict(data_X)
        return cluster_ids

    def ag_max_c(self, data_X):
        clusterer = ag(
                n_clusters=self.num_clusters,
                metric=self.metric,
                linkage='complete')
        cluster_ids = clusterer.fit_predict(data_X)
        return cluster_ids

    def _prepare_clusters(self, cluster_ids, scores):
        clusters = [[] for i in range(self.num_clusters)]
        for i,cluster_id in enumerate(cluster_ids):
            clusters[cluster_id].append(i)
        clusters = sorted(clusters, key=lambda cluster: len(cluster))
        return clusters

    def uniform_s(self, **kwargs):
        selected_idxs = self._residual_sampling(selected_idxs=[], **kwargs)
        return selected_idxs

    def proportional_s(self, select_size):
        selected_idxs = []
        num_pool = sum(len(self.clusters[i]) for i in range(self.num_clusters))
        for i in range(self.num_clusters):
            proportion = float(len(self.clusters[i]))/num_pool
            acq_num = int(proportion*select_size)
            if acq_num>0:
                selected_idxs += self.clusters[i][-acq_num:]
                self.clusters[i] = self.clusters[i][:-acq_num]
        selected_idxs = self._residual_sampling(
                selected_idxs=selected_idxs,
                select_size=select_size)
        return selected_idxs

    def _residual_sampling(self, selected_idxs, select_size):
        cluster_idx = 0
        while len(selected_idxs)<select_size:
            if len(self.clusters[cluster_idx%self.num_clusters])>0:
                idx = self.clusters[cluster_idx%self.num_clusters].pop(-1)
                selected_idxs.append(idx)
            cluster_idx += 1
        return selected_idxs

    def fit(self, data_X, scores, **kwargs):
        cluster_ids = self.clustering(data_X=data_X.numpy())
        clusters = self._prepare_clusters(
                cluster_ids=cluster_ids,
                scores=scores)
        max_len = max([len(c) for c in clusters])
        min_len = min([len(c) for c in clusters])
        avg_len = np.mean([len(c) for c in clusters])
        median_len = np.median([len(c) for c in clusters])
        self.logger.info(
            (f'[num clusters: {len(clusters)}]'
             f'[max: {max_len}][min: {min_len}][avg: {avg_len:.1f}][median: {median_len:.1f}]'))
        self.clusters = clusters

    def __call__(self, select_size):
        assert hasattr(self, 'clusters'), "fit the Cluster diversifier first."
        local_idxs = self.sampling(select_size=select_size)
        return local_idxs