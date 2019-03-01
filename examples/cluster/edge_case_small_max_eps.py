"""
Bench min_samples manually entered values, max_eps in range that depends on the
analytically optimal max_eps.
This can't find my small cluster.
"""

from sklearn.cluster import OPTICS
from sklearn.cluster.optics_ import _compute_core_distances_
import matplotlib.gridspec as gridspec

import os
import numpy as np
import matplotlib.pyplot as plt


# figure output directory
out_dir = 'out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# gen data
n_points_per_cluster = 10
d = 2 # data dimension
np.random.seed(0)

C1 = [0, 0] + 0.05*np.random.randn(n_points_per_cluster, d)
C1[C1>0.05] = 0.05
C1[C1<-0.05] = -0.05

C2 = [50, 50] + 1*np.random.randn(n_points_per_cluster, d)
C2[C2>51] = 51
C2[C2<49] = 49

C3 = [100, 100] + 10* np.random.randn(n_points_per_cluster, d)
C3[C3>110] = 110
C3[C3<90] = 90

X = np.vstack((C1, C2, C3))


grid_id = -1
for min_samples in [3,6,9]:
    
    for max_eps in [0.051, 1.1, 10.1]:
        grid_id += 1 
        fig_name = '%s/%d.png'%(out_dir,grid_id)
        params_str = '%05d, min_samples:%.3f, max_eps:%.3f'
        print(params_str %(grid_id, min_samples, max_eps))

        clust = OPTICS(min_samples=min_samples,
                max_eps = max_eps,
                cluster_method='xi', 
                eps=0.5,
                xi=5e-2,
                min_cluster_size=5e-3,
                leaf_size = 30)
        clust.fit(X)
        
        # get results 
        space = np.arange(len(X))
        reachability = clust.reachability_[clust.ordering_]
        labels = clust.labels_[clust.ordering_]
        #print('labels', labels)
        #print('reachability', reachability)
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
                %(grid_id, min_samples, max_eps, cluster_num))

        # OPTICS
        color = ['g.', 'r.', 'b.', 'y.', 'c.']
        for k, c in zip(range(0, 5), color):
            Xk = X[clust.labels_ == k]
            ax2.plot(Xk[:, 0], Xk[:, 1], c, alpha=0.3)
        ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
        ax2.set_title('Automatic Clustering\nOPTICS')
        
        # plot raw data
        ax3.plot(X[:,0], X[:,1], 'k+', alpha=0.1)
        ax3.set_title('All data')

        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close()
        
