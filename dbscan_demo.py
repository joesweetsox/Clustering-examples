# for data prep, clustering and evaluation
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np

# for plotting
import matplotlib.pyplot as plt

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
    # Compute DBSCAN
    db = DBSCAN(eps=0.175, min_samples=5, metric='euclidean', algorithm='auto', n_jobs=-1).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
        % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, labels))

    # #############################################################################
    # Plot result

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.savefig("dbscan.pdf")
    plt.show()
