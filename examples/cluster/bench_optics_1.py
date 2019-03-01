"""
Play with reachability plot params i.e. keep default params for cluster
extraction.
Bench: 
    - min_samples
    - leaf_size (it seems it has no influence on the cluster result but it may
      have on the time perfs)
    - max eps is computed with the optimal formula of the paper
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
BENCH_ID = 1
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
min_samples_l   = np.linspace(1,30,5).astype(np.int)
max_eps_l         = np.linspace(10,100,10)
#leaf_size_l     = np.linspace(5,30,5).astype(np.int) # useless
hull = ConvexHull(X)

# Params for cluster extraction (default params)
xi    = 0.05 #np.linspace(0.1,1,4)
mr    = 0.75 # maxima ratio
rr    = 0.7 # rejection ratio
st    = 0.4 # sim threshold
mcs   = 5e-3 #np.linspace(0.001,0.01,10) # min cluster size
mmr   = 1e-3 #np.linspace(0.001,0.01,10) # min maxima ratio

# grid search generation
grid_search = []
for ms in min_samples_l:
    for ls in max_eps_l:
    grid_search.append([ms])
grid_search = np.array(grid_search)
grid_id = -1
start_time = time.time()
f = open('desc.txt', 'w')

# let's go
for params in grid_search:
    grid_id += 1
    ms, ls = params
    V_DS = hull.area * X.shape[0] / ms
    max_eps = d * np.sqrt(V_DS * ms * gamma(1.0*d/2 + 1) / (X.shape[0] * np.sqrt(np.pi**d)))

    fig_name = '%s/%d.png'%(RES_DIR,grid_id)
    f.write('%d,%.3f,%.3f,%.3f' %(grid_id, ms, ls, max_eps))

    clust = OPTICS(min_samples=ms, # minPts
            max_eps = max_eps, # for NN
            cluster_method='xi', 
            eps=0.5,
            xi=5e-2,
            min_cluster_size=5e-3,
            leaf_size = ls)
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
    ax1.set_title('%05d, min_samples:%.3f, leaf_size:%.3f, max_eps:%.3f'
            %(grid_id, ms, ls, max_eps))

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
    print( '%05d, min_samples:%.3f, leaf_size:%.3f, max_eps:%.3f - :%d:%02d'
            %(grid_id, ms, ls, max_eps, int(duration/60), duration%60))

f.close()
