from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import silhouette_score
df1 = pd.read_excel('./prav.xlsx')

Ks = range(2, 15)
Ds = []
for K in Ks:
 cls = KMeans(n_clusters=K, random_state=1).fit(df1)
 labels = cls.labels_
 Ds.append(silhouette_score(df1, labels, metric='euclidean'))
plt.plot(Ks, Ds, 'o-')
plt.xlabel(u'Величина K')
plt.ylabel(u'Силуэтная отметка')
plt.show()
