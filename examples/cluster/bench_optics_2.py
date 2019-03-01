"""
Bench min_samples and max_eps with manually entered values.
"""

from sklearn.cluster import OPTICS
import matplotlib.gridspec as gridspec
import time

import os
import numpy as np

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

import cv2
from scipy.spatial import ConvexHull
from scipy.special import gamma

# Constant params
n_jobs = -1
metric = 'minkowski'
p = 2
metric_params = None
algorithm = 'auto'

# Output dir
BENCH_ID = 2
RES_DIR = 'res/bench_%d'%BENCH_ID
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

# Generate sample data
np.random.seed(0)
n_points_per_cluster = 250
d = 2 # data dimension
C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, d)
C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, d)
C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, d)
C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, d)
C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, d)
C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, d)
X = np.vstack((C1, C2, C3, C4, C5, C6))

# Params for reachability plot
min_samples_l   = np.linspace(1,100,11).astype(np.int)
max_eps_l       = np.linspace(1,10,10).astype(np.int)

# grid search generation
grid_search = []
for ms in min_samples_l:
    for me in max_eps_l:
        grid_search.append([ms,me])
grid_search = np.array(grid_search)
grid_id = -1
start_time = time.time()
f = open('desc.txt', 'w')

# optimal max_eps according to the paper
hull = ConvexHull(X)


# let's go
for params in grid_search:
    grid_id += 1
    ms, me = params

    # analytic max eps
    max_eps = hull.area * ms * gamma(1.0*d/2 + 1) / (X.shape[0] * np.sqrt(np.pi**d))
    max_eps = max_eps**(1.0/d)

    fig_name = '%s/%d.png'%(RES_DIR,grid_id)
    params_str = '%05d, min_samples:%.3f, max_eps/opt_max_eps:%.3f/%.3f'
    f.write('%d,%.3f,%.3f,%.3f' %(grid_id, ms, me, max_eps))

    clust = OPTICS(min_samples=ms, # minPts
            max_eps = me, # for NN
            cluster_method='xi', 
            eps=0.5, # for dbscan only
            xi=5e-2,
            min_cluster_size=5e-3,
            leaf_size = 30)
    clust.fit(X)
    #if grid_id==10:
    #    break
    
    # get results 
    space = np.arange(len(X))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    
    # plots stuff
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
    ax1.set_title(params_str %(grid_id, ms, me, max_eps))

    # OPTICS
    color = ['g.', 'r.', 'b.', 'y.', 'c.']
    for k, c in zip(range(0, 5), color):
        Xk = X[clust.labels_ == k]
        ax2.plot(Xk[:, 0], Xk[:, 1], c, alpha=0.3)
    ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    ax2.set_title('Automatic Clustering\nOPTICS')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()
    
    # screen log
    duration = time.time() - start_time
    print( (params_str + ' - :%d:%02d') 
            %(grid_id, ms, me, max_eps, int(duration/60), duration%60))

f.close()
