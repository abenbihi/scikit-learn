"""
Set `max_eps=np.inf` to avoid any edge cases.
Bench `min_samples`, `xi` and `min_cluster_size`
"""

from sklearn.cluster import OPTICS
from sklearn.cluster.optics_ import _compute_core_distances_
import matplotlib.gridspec as gridspec
import time

import os
import numpy as np

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

import cv2
from scipy.spatial import ConvexHull
from scipy.special import gamma

import gen_data

# Output dir
BENCH_ID = 5
RES_DIR = 'res/bench_%d'%BENCH_ID
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

n_points_per_cluster = 100
d = 2 # data dimension
X = gen_data.gen1(d, n_points_per_cluster)

# grid search generation
ms_l    = [9] #np.linspace(10,100,10) # min samples
xi_l    = np.linspace(0.09,0.99,10) # xi
mcs_l   = np.linspace(10,100,10) #[9] #np.linspace(0.01, 0.1, 10) # min cluster size

grid_search = []
for ms in ms_l:
    for xi in xi_l:
        for mcs in mcs_l:
            grid_search.append([ms, xi, mcs])
grid_search = np.array(grid_search)
grid_id = -1
start_time = time.time()


# let's go
for params in grid_search:
    grid_id += 1
    ms, xi, mcs = params
    fig_name = '%s/%d.png'%(RES_DIR,grid_id)
    params_str = '%05d, min_samples:%.3f, xi:%.3f, mcs:%.3f'

    clust = OPTICS(min_samples=int(ms), xi=xi, min_cluster_size=mcs)
    clust.fit(X)
    #if grid_id==10:
    #    break
    
    # get results 
    space = np.arange(len(X))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    cluster_num = len(np.unique(labels)) -1
    
    # plots stuff
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(3, 1)
    ax1 = plt.subplot(G[0, 0])
    ax2 = plt.subplot(G[1, 0])
    ax3 = plt.subplot(G[2, 0])
    
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
    ax1.set_title((params_str + ' clus_num:%d') 
            %(grid_id, ms, xi, mcs, cluster_num))

    # OPTICS
    color = ['g.', 'r.', 'b.', 'y.', 'c.']
    for k, c in zip(range(0, 5), color):
        Xk = X[clust.labels_ == k]
        ax2.plot(Xk[:, 0], Xk[:, 1], c, alpha=0.3)
    ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    ax2.set_title('Automatic Clustering\nOPTICS')

    ax3.plot(X[:,0], X[:,1], 'k+', alpha=0.1)
    ax3.set_title('All data')

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()
    
    # screen log
    duration = time.time() - start_time
    print( (params_str + ' cluster_num: %d - :%d:%02d') 
            %(grid_id, ms, xi, mcs, cluster_num, int(duration/60), 
                duration%60))
    #img = cv2.imread(fig_name) # because fuck mpl
    #cv2.imshow('img', img)
    #cv2.waitKey(0)

