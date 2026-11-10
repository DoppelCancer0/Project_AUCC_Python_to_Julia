"""
    I vari print inseriti da me erano per capire un problema di lettura dell'output del print aucc;
"""


print("Prima dell'esecuzione")

print("Fase Importo")
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
from sklearn.cluster import KMeans

import numpy as np

from aucc import aucc

# As usage example, we use the same code employed for Silhouette in scikit-learn
# available at: https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
print("Fase Creazione variabili")
X, y = datasets.load_linnerud(return_X_y=True)
kmeans_model = KMeans(n_clusters=3, random_state=1, n_init='auto').fit(X)
labels = kmeans_model.labels_
print("Fase Stampa")
print(aucc(X, labels, metric='euclidean'))
print("Fine Esecuzione")