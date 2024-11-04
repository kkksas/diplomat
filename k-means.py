from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
df1 = pd.read_excel('./prav.xlsx')
#оно дает слишком большой вклад 
#df1['время трансфера'] = df1['время трансфера'] / 24
"""
Ks = range(2, 15)
Ds = []
inertia = []
for K in Ks:
 cls = KMeans(n_clusters=K, random_state=1).fit(df1)
 labels = cls.labels_
 inertia.append(cls.inertia_)
 Ds.append(silhouette_score(df1, labels, metric='euclidean'))
plt.plot(Ks, Ds, 'o-')
plt.xlabel(u'Величина K')
plt.ylabel(u'Силуэтная отметка')
plt.show()
"""
def silhouette(N_clusters):
    cls = KMeans(n_clusters=N_clusters, random_state=1).fit(df1)
    return silhouette_score(df1, cls.labels_, metric='euclidean')


def closeness(N_clusters):
    cls = KMeans(n_clusters=N_clusters, random_state=1).fit(df1)
    mean_inter_claster=[]
    for i in range(N_clusters):
        mask = cls.labels_==i
        labeledRec = df1[mask]
        dist = euclidean_distances(labeledRec, labeledRec)
        sum_dist = np.sum(dist, axis=1)
        mean_dist = sum_dist / (dist.size-1)
        mean_inter_claster.append(np.mean(mean_dist))
    a = np.mean(mean_inter_claster, axis=0)
    print(a)
    return a 

def farness(N_clusters):
    cls = KMeans(n_clusters=N_clusters, random_state=1).fit(df1)
    nearest_cluster_distance = []
    for i in range(N_clusters):        
        mask = cls.labels_==i
        labeledRec = df1[mask].to_numpy()
        for j in labeledRec:
            foreign_clusters = np.delete(cls.cluster_centers_, i,0)
            second_closest_cluster = foreign_clusters[np.argmin(euclidean_distances(j.reshape(1, -1), foreign_clusters),axis=1)]
            index=-1
            for k in range(N_clusters):
                if (cls.cluster_centers_[k]== second_closest_cluster).all():
                    index = k 
                    break
            scc_mask = cls.labels_==index
            scc_align_rec = df1[scc_mask].to_numpy()
            nearest_cluster_distance.append(np.mean(euclidean_distances(j.reshape(1, -1), scc_align_rec))) 
    return np.mean(nearest_cluster_distance)           

print(silhouette(2))
print(My_silhouette(2))