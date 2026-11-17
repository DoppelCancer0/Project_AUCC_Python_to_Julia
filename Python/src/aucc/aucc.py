"""Imports a function to check if data and labels have the same size"""
from sklearn.utils import check_X_y
"""Imports a function to convert cluster labels into integers"""
from sklearn.preprocessing import LabelEncoder
"""Imports a function to check if the cluster's number is valid"""
from sklearn.metrics.cluster._unsupervised import check_number_of_labels
"""Imports a function to calculate the area under the ROC curve"""
from sklearn.metrics import roc_auc_score

"""Imports a function that calculates the distance between two elements of a couple"""
from scipy.spatial.distance import pdist

"""Imports math operations and numerical arrays"""
import numpy as np

def aucc(X, labels, metric='euclidean'):
    """Computes AUCC (Area Under the (ROC) Curve for Clustering).

    Maximum score is 1.0. The higher the score, the better the partition.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    
    metric : distance employed between data points. Default is euclidean.

    Returns
    -------
    score: float
        The resulting AUCC score.

    References
    ----------
    .. [1] Pablo A. Jaskowiak, Ivan G. Costa, and Ricardo J. G. B. Campello. 2022. 
    The area under the ROC curve as a measure of clustering quality. 
    Data Min. Knowl. Discov. 36, 3 (May 2022), 1219–1245. 
    https://doi.org/10.1007/s10618-022-00829-0
    """

    """Checks if X's entries are equal to labels' entries and returns the same values to the respective variables"""
    X, labels = check_X_y(X, labels)
    """Creates a LabelEncoder object that transforms labels into integers"""
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    """Returns number of rows from X for n_samples, and number of columns is voided"""
    n_samples, _ = X.shape
    """le.classes contains all the integers assigned to all the different labels;
       len(...) counts the integers and assigns the number to n_labels"""
    n_labels = len(le.classes_)
    """Checks if number of labels and number of samples is valid for the operation;
       specifically, the number of labels (clusters) needs to be strictly bigger than 1,
       and the number of labels must be strictly lower than the number of samples"""
    check_number_of_labels(n_labels, n_samples)

    """Calculates the number of pairs from the number of samples;
       the division is an integer division.
       """
    n_pairs = n_samples*(n_samples-1)//2

    """If the labels Matrix has a single row, calculate pairwise_labels with the transposed matrix."""
    if np.matrix(labels).shape[0] == 1:
        """Hamming Distance: distance calculated by the amount of different pairs between two vectors;
           pdist calculates the hamming distance between each possible pair of the (transposed) labels' matrix;
           then 1 is subtracted by the vector of hamming distances, and it becomes the pairwise_labels"""
        pairwise_labels    = 1 - pdist(np.matrix(labels).transpose(),'hamming')
    else:
        pairwise_labels    = 1 - pdist(np.matrix(labels),'hamming')
    """Calculates the distance between each point through the euclidean expression, and allocates them in a vector."""
    tmpdist = pdist(X,metric)
    tmpdist = (tmpdist - np.min(tmpdist))/(np.max(tmpdist) - np.min(tmpdist))
    pairwise_distances = 1 - tmpdist

    return roc_auc_score(pairwise_labels,pairwise_distances)
