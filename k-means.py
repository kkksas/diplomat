from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances
df1 = pd.read_excel('./prav.xlsx')
#df1.reset_index()
#df1 = df1.drop('Unnamed: 0')
print(df1)
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

Ks = range(2,13)
Ds = []
Ds2 = []
Ds3 = []
ases= []
inertia = []
for k in Ks:
    cls = KMeans(k,random_state=1).fit(df1)
    #labels = np.array(cls.labels_)
    #labels = labels[labels!=-1]
    #Ks.append(len(set(labels)))
    Ds.append(silhouette_score(df1, cls.labels_, metric='euclidean'))
    Ds2.append(davies_bouldin_score(df1, cls.labels_))
    Ds3.append(calinski_harabasz_score(df1, cls.labels_))
'''
df1 = pd.DataFrame({'ks':Ks, 'ds':Ds})
df1 = df1.groupby('ks').max()
df2 = pd.DataFrame({'ks':Ks, 'ds':Ds2})
df2 = df2.groupby('ks').min()
df3 = pd.DataFrame({'ks':Ks,'ds':Ds3})
df3 = df3.groupby('ks').max()
'''

plt.subplot(1, 3, 1)
plt.plot( Ks,Ds, 'o-')
#plt.plot(Ks, Ds2, 'x-')
plt.title('Коэффициент силуэта \n при различном количестве кластеров', fontsize=16)
plt.xlabel(u'Количество кластеров',fontsize=14)
plt.ylabel(u'Коэффициент силуэта',fontsize=14)

plt.subplot(1, 3, 2)
plt.plot( Ks, Ds2, 'o-')
#plt.plot(Ks, Ds2, 'x-')
plt.title('Индекс Дэвиса-Булдина \n при различном количестве кластеров',fontsize=16)
plt.xlabel(u'Количество кластеров',fontsize=14)
plt.ylabel(u'Индекс Дэвиса-Булдина',fontsize=14)

plt.subplot(1, 3, 3)
plt.plot( Ks,Ds3, 'o-')
#plt.plot(Ks, Ds2, 'x-')
plt.title('Индекс Калинского-Харабаша \n при различном количестве кластеров',fontsize=16)
plt.xlabel(u'Количество кластеров',fontsize=14)
plt.ylabel(u'Индекс Калинского-Харабаша',fontsize=14)
plt.show()


"""
plt.plot(Ks, ases, 'o-')
plt.title('Средняя связность объектов в класстере')
plt.xlabel(u'Количество кластеров')
plt.ylabel(u'Связность')


cls = KMeans(n_clusters=20, random_state=1).fit(df1)
labels = cls.labels_
mySel(20, cls)"""