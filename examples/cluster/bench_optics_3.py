"""
Bench min_samples manually entered values, max_eps in range that depends on the
analytically optimal max_eps. 
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

# Constant params
n_jobs = -1
metric = 'minkowski'
p = 2
metric_params = None
algorithm = 'auto'
LEAF_SIZE = 30

# Output dir
BENCH_ID = 3
RES_DIR = 'res/bench_%d'%BENCH_ID
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

n_points_per_cluster = 10
d = 2 # data dimension
X = gen_data.gen4(d, n_points_per_cluster)

# grid search generation
grid_search = []
for ms in [3]: 
    grid_search.append([ms])
grid_search = np.array(grid_search)
grid_id = -1
start_time = time.time()
f = open('desc.txt', 'w')

# optimal max_eps according to the paper
hull = ConvexHull(X)


# let's go
for params in grid_search:
    ms = params[0]
    
    # analytic optimal max eps
    max_eps_an = hull.area * ms * gamma(1.0*d/2 + 1) / (X.shape[0] * np.sqrt(np.pi**d))
    max_eps_an = max_eps_an**(1.0/d)

    # xp optimal max_eps
    nbrs = NearestNeighbors(n_neighbors=ms, leaf_size=LEAF_SIZE)
    nbrs.fit(X)
    core_distances_ = _compute_core_distances_(X=X, neighbors=nbrs,
                                               min_samples=ms,
                                               working_memory=None)
    max_eps_xp = np.mean(core_distances_)

    # choose which max_eps to use
    if max_eps_xp==0:
        print('warning: max_eps_xp=0. Use the analytical one.')
        max_eps = max_eps_an
    else:
        max_eps = max_eps_xp
    #print('max_eps_xp', max_eps_xp)
    
    print('Try max_eps in [%.3f,%.3f]'%(max_eps/10, 10*max_eps))
    print('\nmin_samples: %d, max_eps_an:%.3f, max_eps_xp:%.3f'
            %(ms,max_eps_an,max_eps_xp))

    # let's go
    #for me in np.linspace(max_eps/10, 10*max_eps, 20):
    for me in [0.051, 1.1, 10.1]:
        grid_id += 1

        fig_name = '%s/%d.png'%(RES_DIR,grid_id)
        params_str = '%05d, min_samples:%.3f, max_eps:%.3f/%.3f'
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
        print(labels)
        print(reachability)
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
                %(grid_id, ms, me, max_eps, cluster_num))

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
                %(grid_id, ms, me, max_eps, cluster_num, int(duration/60), 
                    duration%60))
        img = cv2.imread(fig_name) # because fuck mpl
        cv2.imshow('img', img)
        cv2.waitKey(0)

f.close()
