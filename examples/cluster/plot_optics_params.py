# -*- coding: utf-8 -*-
"""
===================================
Demo of OPTICS clustering algorithm
===================================

Finds clusters of different densities.

"""
print(__doc__)

import numpy as np

from sklearn.cluster import OPTICS
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


n_points_per_cluster = 100
d = 2 # data dimension


# #############################################################################
# Generate sample data
np.random.seed(0)
centers = [[-5, -2], [4, -1], [1, -2], [-2, 3], [3, -2], [5, 6]]
variances = [.8, .1, .2, .3, 1.6, 2]
blobs = []
labels_true = []
for i,c in enumerate(centers):
    blobs.append(c + variances[i]*np.random.randn(n_points_per_cluster, d))
    labels_true.append(np.ones(n_points_per_cluster)*i)
X = np.vstack(blobs)
labels_true = np.hstack(labels_true).astype(np.int)


# #############################################################################
# Compute OPTICS


# 1. Increase `min_samples` until the steepness plot is smoothed
# and has peaks only at the breaks in the reachability plot.
print("Set min_samples. Fix xi=%.2f")
xi = 0.55
# plots stuff
plt.figure(1, figsize=(15, 9))
G = gridspec.GridSpec(4, 3)
for i, min_samples in enumerate([1, 16, 80]):
    min_cluster_size = min_samples
    params_str = 'min_samples:%d'
    clust = OPTICS(min_samples=min_samples, xi=xi,
            min_cluster_size=min_cluster_size)
    clust.fit(X)

    # get results 
    space = np.arange(len(X))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    cluster_num = len(np.unique(labels)) -1
    
    # compute steepness
    reach_p1 = np.roll(reachability, 1)[1:]
    steep_up = reachability[1:] / reach_p1
    steep_down = reach_p1 / reachability[1:]
    xi_steep = np.where(steep_up>xi)

    # core distance stat
    core_d = clust.core_distances_[clust.ordering_]

    # plots 
    ax1 = plt.subplot(G[0, i])
    ax2 = plt.subplot(G[1, i])
    ax3 = plt.subplot(G[2, i])
    ax4 = plt.subplot(G[3, i])
    
    # Reachability plot
    color = ['g.', 'r.', 'b.', 'y.', 'c.']
    for k, c in zip(range(0, 5), color):
        Xk = space[labels == k]
        Rk = reachability[labels == k]
        ax1.plot(Xk, Rk, c, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax1.set_title((params_str + '. Cluster number:%d') 
            %(min_samples, cluster_num))
    ax1.plot(space, core_d, 'm', alpha=0.3)
    if i ==0:
        ax1.set_ylabel('Reachability (epsilon distance)')

    # Steepness plot
    x_len = steep_up.shape[0]
    x_axis = np.arange(x_len)
    ax2.plot(x_axis, steep_up, 'r', label='up', alpha=0.5)
    ax2.plot(x_axis, steep_down, 'b', label='down', alpha=0.5)
    #ax2.plot(x_axis, np.ones(x_len)*(1/(1-xi)), 'g', label='xi', alpha=1)
    ax2.plot(x_axis, np.ones(x_len)*(1/(1-xi)), 'g', label='1/(1-xi)=%.1f' 
            %(1/(1-xi)), alpha=1)
    ax2.legend(loc=1, prop={'size': 6})
    if i==0:
        ax2.set_ylabel('Steepness')
    if i==1:
        ax2.set_title('Steepness plot. 1/(1-xi)=%.3f'%(1/(1-xi)))

    # OPTICS
    color = ['g.', 'r.', 'b.', 'y.', 'c.']
    for k, c in zip(range(0, 5), color):
        Xk = X[clust.labels_ == k]
        ax3.plot(Xk[:, 0], Xk[:, 1], c, alpha=0.3)
    ax3.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    if i==1:
        ax3.set_title('Automatic Clustering\nOPTICS')
    
    ax4.plot(X[:,0], X[:,1], 'k+', alpha=0.1)
    if i==1:
        ax4.set_title('All data')
    
plt.tight_layout()
plt.title(r'Setting min_samples. (xi=%.2f, $\frac{1}{1-x}$ = %.2f)' %(xi,
    1/(1-xi)))

plt.show()
#plt.close()
print('A suitable value for min_samples is 16')


###############################################################################
# 2. Set Increase `min_samples` until the steepness plot is smoothed
# and has peaks only at the breaks in the reachability plot.
print("Set xi. Fix: min_samples=min_cluster_size=16")
plt.figure(2, figsize=(15, 9))
G = gridspec.GridSpec(4, 3)
plt.title('Setting xi. min_samples=%d'%min_samples)
min_samples = 16
min_cluster_size = min_samples
for i, xi in enumerate([0.0001, 0.1, 0.9]):
    params_str = 'xi=%.3f'
    clust = OPTICS(min_samples=min_samples, xi=xi,
            min_cluster_size=min_samples)
    clust.fit(X)

    # get results 
    space = np.arange(len(X))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    cluster_num = len(np.unique(labels)) -1
    
    # compute steepness
    reach_p1 = np.roll(reachability, 1)[1:]
    steep_up = reachability[1:] / reach_p1
    steep_down = reach_p1 / reachability[1:]
    xi_steep = np.where(steep_up>xi)

    # core distance stat
    core_d = clust.core_distances_[clust.ordering_]

    # plots 
    ax1 = plt.subplot(G[0, i])
    ax2 = plt.subplot(G[1, i])
    ax3 = plt.subplot(G[2, i])
    ax4 = plt.subplot(G[3, i])
    
    # Reachability plot
    color = ['g.', 'r.', 'b.', 'y.', 'c.']
    for k, c in zip(range(0, 5), color):
        Xk = space[labels == k]
        Rk = reachability[labels == k]
        ax1.plot(Xk, Rk, c, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax1.set_title((params_str + '. Cluster number:%d') 
            %(xi, cluster_num))
    ax1.plot(space, core_d, 'm', alpha=0.3)
    if i ==0:
        ax1.set_ylabel('Reachability (epsilon distance)')

    # Steepness plot
    x_len = steep_up.shape[0]
    x_axis = np.arange(x_len)
    ax2.plot(x_axis, steep_up, 'r', label='up', alpha=0.5)
    ax2.plot(x_axis, steep_down, 'b', label='down', alpha=0.5)
    ax2.plot(x_axis, np.ones(x_len)*(1/(1-xi)), 'g', label='1/(1-xi)=%.1f' 
            %(1/(1-xi)), alpha=1)
    ax2.legend(loc=1, prop={'size': 6})
    if i==0:
        ax2.set_ylabel('Steepness')
    ax2.set_title('Steepness plot. 1/(1-xi)=%.3f'%(1/(1-xi)))

    # OPTICS
    color = ['g.', 'r.', 'b.', 'y.', 'c.']
    for k, c in zip(range(0, 5), color):
        Xk = X[clust.labels_ == k]
        ax3.plot(Xk[:, 0], Xk[:, 1], c, alpha=0.3)
    ax3.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    if i==1:
        ax3.set_title('Automatic Clustering\nOPTICS')
    
    ax4.plot(X[:,0], X[:,1], 'k+', alpha=0.1)
    if i==1:
        ax4.set_title('All data')
    
plt.tight_layout()
plt.show()
#plt.close()
print('A suitable value for xi is 0.1')

###############################################################################
# 3. Metrics
min_samples = 16
xi = 0.1
min_cluster_size = min_samples
clust = OPTICS(min_samples=min_samples, xi=xi,
            min_cluster_size=min_cluster_size)
clust.fit(X)
# get results 
space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels_ordered = clust.labels_[clust.ordering_]
labels = clust.labels_
cluster_num = len(np.unique(labels)) -1

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('\n\nEstimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels,
                                           average_method='arithmetic'))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# plots stuff
plt.figure(3, figsize=(15, 9))
G = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(G[0, 0])

# OPTICS
color = ['g.', 'r.', 'b.', 'y.', 'c.']
for k, c in zip(range(0, 5), color):
    Xk = X[clust.labels_ == k]
    ax1.plot(Xk[:, 0], Xk[:, 1], c, alpha=0.3, markersize=10)
ax1.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.3,
        markersize=10)
ax1.set_title('Automatic Clustering\nOPTICS')

plt.tight_layout()
plt.show()
#plt.close()
