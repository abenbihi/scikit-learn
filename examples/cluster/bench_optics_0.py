"""
===================================
Demo of OPTICS clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.
This example uses data that is generated so that the clusters have
different densities.

The clustering is first used in its automatic settings, which is the
:class:`sklearn.cluster.OPTICS` algorithm, and then setting specific
thresholds on the reachability, which corresponds to DBSCAN.

We can see that the different clusters of OPTICS can be recovered with
different choices of thresholds in DBSCAN.

"""

# Authors: Shane Grigsby <refuge@rocktalus.com>
#          Amy X. Zhang <axz@mit.edu>
# License: BSD 3 clause


from sklearn.cluster import OPTICS
import matplotlib.gridspec as gridspec
import time


import numpy as np

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

import cv2

# constant params
n_jobs = -1
metric = 'minkowski'
p = 2
metric_params = None
algorithm = 'auto'

# Generate sample data

np.random.seed(0)
n_points_per_cluster = 250

C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
X = np.vstack((C1, C2, C3, C4, C5, C6))

# core objects dfn
MIN_SAMPLES = 5
MAX_EPS = np.inf
LEAF_SIZE = 30

# extract cluster
xi_l    = np.linspace(0.1,1,4)
mr_l    = np.linspace(0.1,1,4) # maxima ratio
rr_l    = np.linspace(0.1,1,4) # rejection ratio
st_l    = np.linspace(0.1,1,4) # sim threshold
sm_l    = np.linspace(0.1,1,4) # significant min
mcs_l   = np.linspace(0.1,1,4) # min cluster size
mmr_l   = np.linspace(0.1,1,4) # min maxima ratio

leaf_size = 30 

grid_search = []
for xi in xi_l:
    for mr in mr_l:
        for rr in rr_l:
            for st in st_l:
                for sm in sm_l:
                    for mcs in mcs_l:
                        for mmr in mmr_l:
                            grid_search.append(
                                    [xi, mr, rr, st, sm, mcs, mmr])

grid_search = np.array(grid_search)

clust = OPTICS(min_samples=MIN_SAMPLES, 
            max_eps = MAX_EPS, 
            leaf_size = LEAF_SIZE,
            extract_method='xi', 
            xi=xi,
            maxima_ratio=mr, 
            rejection_ratio=rr,
            similarity_threshold=st,
            significant_min=sm, 
            min_cluster_size=mcs,
            min_maxima_ratio=mmr)

# do the neighboring computation only once
#X = check_array(X, dtype=np.float)

n_samples = len(X)

if clust.min_samples > n_samples:
    raise ValueError("Number of training samples (n_samples=%d) must "
                     "be greater than min_samples (min_samples=%d) "
                     "used for clustering." %
                     (n_samples, clust.min_samples))

if clust.min_cluster_size <= 0 or (clust.min_cluster_size !=
                                  int(clust.min_cluster_size)
                                  and clust.min_cluster_size > 1):
    raise ValueError('min_cluster_size must be a positive integer or '
                     'a float between 0 and 1. Got %r' %
                     clust.min_cluster_size)
elif clust.min_cluster_size > n_samples:
    raise ValueError('min_cluster_size must be no greater than the '
                     'number of samples (%d). Got %d' %
                     (n_samples, clust.min_cluster_size))

if clust.extract_method not in ['dbscan', 'sqlnk', 'xi']:
    raise ValueError("extract_method should be one of"
                     " 'dbscan', 'xi', or 'sqlnk', but is %s" %
                     clust.extract_method)

# Start all points as 'unprocessed' ##
clust.reachability_ = np.empty(n_samples)
clust.reachability_.fill(np.inf)
clust.predecessor_ = np.empty(n_samples, dtype=int)
clust.predecessor_.fill(-1)
# Start all points as noise ##
clust.labels_ = np.full(n_samples, -1, dtype=int)

nbrs = NearestNeighbors(n_neighbors=clust.min_samples,
                        algorithm=clust.algorithm,
                        leaf_size=clust.leaf_size,
                        metric=clust.metric,
                        metric_params=clust.metric_params,
                        p=clust.p,
                        n_jobs=clust.n_jobs)
nbrs.fit(X)


grid_id = -1
start_time = time.time()
f = open('desc.txt', 'w')
for params in grid_search:
    grid_id += 1
    xi, mr, rr, st, sm, mcs, mmr = params
    fig_name = 'res/%d.png'%grid_id
    f.write('%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f'
            %(grid_id, xi, mr, rr, st, sm, mcs, mmr))

    clust.xi                    = xi
    clust.maxima_ratio          = mr
    clust.rejection_ratio       = rr
    clust.similarity_threshold  = st
    clust.significant_min       = sm
    clust.min_cluster_size      = mcs
    clust.min_maxima_ratio      = mmr
    
    
    clust.core_distances_ = clust._compute_core_distances_(X, nbrs)
    # OPTICS puts an upper limit on these, use inf for undefined.
    clust.core_distances_[clust.core_distances_ > clust.max_eps] = np.inf

    clust.ordering_ = clust._calculate_optics_order(X, nbrs)
    labels_ = clust.extract_xi(clust.xi, False)
    indices_ = None

    clust.core_sample_indices_ = indices_
    clust.labels_ = labels_

    _, labels_025 = clust.extract_dbscan(0.25)
    _, labels_075 = clust.extract_dbscan(0.75)
    
    space = np.arange(len(X))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 1)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])
    
    # Reachability plot
    color = ['g.', 'r.', 'b.', 'y.', 'c.']
    for k, c in zip(range(0, 5), color):
        Xk = space[labels == k]
        Rk = reachability[labels == k]
        ax1.plot(Xk, Rk, c, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax1.plot(space, np.full_like(space, 0.75, dtype=float), 'k-', alpha=0.5)
    ax1.plot(space, np.full_like(space, 0.25, dtype=float), 'k-.', alpha=0.5)
    ax1.set_ylabel('Reachability (epsilon distance)')

    params_str = '%04d, xi:%.2f, max_rat:%.2f, rej_rat:%.2f, sim_thresh:%.2f,'
    params_str += ' sign_min:%.2f, min_clus_sz:%.2f, min_max_rat:%.2f '
    ax1.set_title(params_str %(grid_id, xi, mr, rr, st, sm, mcs, mmr))

    # OPTICS
    color = ['g.', 'r.', 'b.', 'y.', 'c.']
    for k, c in zip(range(0, 5), color):
        Xk = X[clust.labels_ == k]
        ax2.plot(Xk[:, 0], Xk[:, 1], c, alpha=0.3)
    ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    ax2.set_title('Automatic Clustering\nOPTICS')
    
    plt.tight_layout()
    #plt.show()

    plt.savefig(fig_name)
    plt.close()
    
    duration = time.time() - start_time
    print( (params_str + ' %d:%02d')
            %(grid_id, clust.xi, clust.maxima_ratio, clust.rejection_ratio,
                clust.similarity_threshold, clust.significant_min,
                clust.min_cluster_size, clust.min_maxima_ratio, int(duration/60),
                duration%60))

f.close()
