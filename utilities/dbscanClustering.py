import numpy as np
from sklearn.cluster import DBSCAN


# Takes image and performs DB scan, returns clustered image and clusters
def perform_dbscan(canny_image, eps=50, min_samples=3, metric='euclidean', algorithm='auto'):
    input_copy = np.copy(canny_image)
    nonzero = input_copy.nonzero()
    nonzero_indices = np.transpose(nonzero)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm)
    db.fit(nonzero_indices)
    labels = db.labels_

    i = 0
    for item in nonzero_indices:
        # Cluster noise points get -1, valid clusters get numbers 0, 1 .. etc.
        # Adding +1 to labels makes our image noise pixels as 0 (hence not seen
        # in image and all other pixels get different colors
        # labels[i] + 1 is done because when we are finding clusters on an image cluster value number as 0 would be considered as empty
        input_copy[item[0], item[1]] = labels[i] + 1
        i = i + 1

    unique, counts = np.unique(labels, return_counts=True)
    unique = np.append(unique, (max(unique)+1))
    unique = np.delete(unique, 0)
    return input_copy, unique, counts
