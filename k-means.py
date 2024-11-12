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
def silhouette(cls):
    #cls = KMeans(n_clusters=N_clusters, random_state=1).fit(df1)
    return silhouette_score(df1, cls.labels_, metric='euclidean')

def closeness2(N_clusters, cls):
    #cls = KMeans(n_clusters=N_clusters, random_state=1).fit(df1)
    mean_inter_claster=[]
    iter = 0 
    for i in df1.to_numpy():
        mask = cls.labels_==cls.labels_[iter]
        labeledRec = df1[mask].to_numpy()
        dist = euclidean_distances(i.reshape(1, -1), labeledRec)
        sum_dist = np.sum(dist, axis=1)
        if dist.size != 1:
            mean_dist = sum_dist / (dist.size-1)
        else:
            mean_dist = 0
        mean_inter_claster = np.append(mean_inter_claster, mean_dist)
        iter += 1
    return mean_inter_claster
        

def farness2(N_clusters, cls):
    #cls = KMeans(n_clusters=N_clusters, random_state=1).fit(df1)
    nearest_cluster_distance = []
    labeledRec = []
    for i in range(N_clusters):
        mask = cls.labels_ == i 
        labeledRec.append(df1[mask].to_numpy())
    for i in df1.to_numpy():
        cluster_dists = []
        leng = len(labeledRec)
        for j in range(leng):
            clust_rec = labeledRec[j]
            eu = euclidean_distances(i.reshape(1, -1), clust_rec)
            if (0 in eu):
                cluster_dists= np.append(cluster_dists, 0)
                continue
            cluster_dists =np.append(cluster_dists, np.mean(eu))
        nearest_cluster_distance = np.append(nearest_cluster_distance, min(cluster_dists[cluster_dists != 0]))
    return nearest_cluster_distance

def mySel(N_clusters, cls):
    a1=closeness2(N_clusters,cls)
    b1=farness2(N_clusters,cls)
    return np.mean((b1-a1)/np.maximum(a1,b1)) 

Ks = range(2, 12)
Ds = []
Ds2 = []
ases= []
inertia = []
for K in Ks:
 cls = KMeans(n_clusters=K, random_state=1).fit(df1)
 labels = cls.labels_
 ases.append(np.mean(closeness2(K,cls)))
 #Ds2.append(mySel(K, cls))
 #Ds.append(silhouette_score(df1, labels, metric='euclidean'))
 
"""
plt.plot(Ks, Ds, 'o-')
plt.plot(Ks, Ds2, 'x-')
plt.title('Силуэтная отметка при различном количестве кластеров')
plt.xlabel(u'Количество кластеров')
plt.ylabel(u'Силуэтная отметка')
plt.legend(['Силуэт из библиотеки sklearn','Силуэт созданный вручную'])
"""
plt.plot(Ks, ases, 'o-')
plt.title('Средняя связность объектов в класстере')
plt.xlabel(u'Количество кластеров')
plt.ylabel(u'Связность')
plt.show()

cls = KMeans(n_clusters=20, random_state=1).fit(df1)
labels = cls.labels_
mySel(20, cls)