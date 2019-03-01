
import numpy as np

def gen1(d, n_points_per_cluster):
    # Generate sample data
    np.random.seed(0)
    C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, d)
    C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, d)
    C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, d)
    C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, d)
    C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, d)
    C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, d)
    X = np.vstack((C1, C2, C3, C4, C5, C6))
    return X

def gen2(d, n_points_per_cluster):
    # Generate sample data
    np.random.seed(0)
    
    
    cluster_num = 10
    mean_l = [ [1,3], [2,4], [6,4], [-5,-2], [0,0], 
            [9,1], [8,3], [3,3], [4,2], [5,5]]
    var_l = np.linspace(0.1, 1, 10)
    C_l = []
    for i in range(cluster_num):
        mean = mean_l[i]
        C_l.append( mean + var_l[i]*np.random.randn(n_points_per_cluster, d))

    X = np.vstack(C_l)
    return X

def gen3(d, n_points_per_cluster):
    # Generate sample data
    np.random.seed(0)
    
    cluster_num = 3
    mean_l = np.random.randint(0,10, (cluster_num, d))
    var_l = np.linspace(0.1, 1, 10)
    C_l = []
    for i in range(cluster_num):
        mean = mean_l[i,:]
        C_l.append( mean + var_l[i]*np.random.randn(n_points_per_cluster, d))

    X = np.vstack(C_l)
    return X

def gen4(d, n_points_per_cluster):
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
    return X

def gen5(d, n_points_per_cluster):
    np.random.seed(0)
    
    # cluster of radius 0.1
    C1 = [[0, 0], [0, 0.1], [0,-.1]]
    
    # cluster of radius 1 center on (10,10)
    C2 = [[10,10], [10,9], [10,11]] 
    
    # cluster of radius 10 centered on (100,100)
    C3 = [[100, 100], [100,90], [100,110]]

    X = np.vstack((C1, C2, C3))
    return X

