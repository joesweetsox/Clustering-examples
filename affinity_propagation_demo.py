# for data prep, clustering and evaluation
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import numpy as np

# for plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def create_data(nobs, vert_scale, horz_scale):
    t = np.linspace(0,2*np.pi,nobs)
    
    g1_x = np.array(horz_scale*np.cos(t)*np.random.uniform(.5,1,len(t)))
    g1_y = np.array(vert_scale*np.sin(t)*np.random.uniform(.5,1,len(t)))
    
    g2_x = np.array(horz_scale*np.cos(t)*np.random.uniform(0,.4,len(t)))
    g2_y = np.array(vert_scale*np.sin(t)*np.random.uniform(0,.4,len(t)))
    
    g3_x = np.array(np.cos(t)*np.random.uniform(0,.6,len(t))-2)
    g3_y = np.array(np.sin(t)*np.random.uniform(0,.8,len(t))-1)
    
    g4_x = np.array(np.cos(t*4)*.5+2+np.random.uniform(-.3,.3,len(t)))
    g4_y = np.array(t/(4*np.pi)*8-2)
    
    x_data = g1_x.tolist()+g2_x.tolist()+g3_x.tolist()+g4_x.tolist()
    y_data = g1_y.tolist()+g2_y.tolist()+g3_y.tolist()+g4_y.tolist()
    return (np.array([list(a) for a in zip(x_data, y_data)]), [0 for a in range(len(g1_x))] + [1 for a in range(len(g2_x))] + [2 for a in range(len(g3_x))] + [3 for a in range(len(g4_x))])

if __name__ == '__main__':
    # #############################################################################
    # Generate sample data
    X, labels_true = create_data(500,3,1)
    # #############################################################################
    # Compute Affinity Propagation
    af = AffinityPropagation(max_iter=500000, convergence_iter=2, random_state=1, verbose=True).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
        % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

    # #############################################################################
    # Plot result

    plt.close('all')
    plt.figure(1)
    plt.clf()

    colors = cm.rainbow(np.linspace(0, 1, n_clusters_))
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        plt.plot(X[class_members, 0], X[class_members, 1], '.', c=col)
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=14)
        for x in X[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], c=col)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.savefig("affinity-propagation.pdf")
    plt.show()
