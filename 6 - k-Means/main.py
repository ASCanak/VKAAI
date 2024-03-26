import numpy as np
import random
from math import dist
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import Counter

def createDataset(fileName):
  """this function transforms a csv into a dataset list."""
  return np.genfromtxt(fileName, delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7],
                       converters={5: lambda s: 0 if s == b"-1" else float(s),
                                   7: lambda s: 0 if s == b"-1" else float(s)})

def normalizeDataset(dataset):
  """Normalize the input dataset using Min-Max normalization."""
  min_vals = np.min(dataset, axis=0)
  max_vals = np.max(dataset, axis=0)
  return (dataset - min_vals) / (max_vals - min_vals)  # Normalized dataset.

def mapDatesToLabels(fileName):
    """
    Maps dates to corresponding labels.

    Parameters:
    - dates: Numpy array, containing date values.

    Returns:
    List of labels based on date ranges.
    """
    dates = np.genfromtxt(fileName, delimiter=';', usecols=[0])

    labels = []
    for label in dates:
        if (label < 20000301) or (20010000 < label < 20010301):
            labels.append('winter')
        elif (20000301 <= label < 20000601) or (20010301 <= label < 20010601):
            labels.append('lente')
        elif (20000601 <= label < 20000901) or (20010601 <= label < 20010901):
            labels.append('zomer')
        elif (20000901 <= label < 20001201) or (20010901 <= label < 20011201):
            labels.append('herfst')
        else: # from 01-12 to end of year
            labels.append('winter')
    return labels


def getLabels(clusters, data, labels):
    """
    Assigns labels to clusters based on data points in each cluster.

    Parameters:
    - clusters: List of lists, each inner list contains indices of data points in a cluster.
    - data: Numpy array, the original data.
    - labels: List, labels corresponding to the data points.

    Returns:
    List of lists, each inner list contains labels for data points in a cluster.
    """
    labelsList = [[] for _ in range(len(clusters))]

    for clusterIndex, cluster in enumerate(clusters):
        for datapoint in cluster:
            labelsList[clusterIndex].append(labels[datapoint])

    return labelsList

def computeCentroids(data, randomCentroids, k):
    """
    Computes centroids and clusters based on randomly selected centroids.

    Parameters:
    - data: Numpy array, the original data.
    - randomCentroids: List, randomly selected centroids.
    - k: Integer, the number of clusters.

    Returns:
    Tuple of lists - Updated centroids and clusters.
    """
    clusters = [[] for _ in range(k)]
    centroids = []

    for datapointIndex, datapoint in enumerate(data):
        distances = [dist(datapoint, centroid) for centroid in randomCentroids]
        closestCentroidIndex = distances.index(min(distances))
        clusters[closestCentroidIndex].append(datapointIndex)

    for cluster in clusters:
        centroids.append(np.mean(data[cluster], axis=0))

    newCentroids = deepcopy(centroids)

    if np.array_equal(centroids, newCentroids):
        return centroids, clusters
    else:
        return computeCentroids(data, newCentroids, k)

def generateScreePlot(k, data):
    """
    Generates a scree plot for different numbers of clusters (K) based on average distance per centroid.

    Parameters:
    - k: Integer, maximum number of clusters to consider.
    - data: Numpy array, the original data.

    Returns:
    None (displays the plot).
    """
    xValues = []
    yValues = []

    for clusterCount in range(k):
        randomCentroids = random.sample(list(data), clusterCount + 1)
        updatedCentroids, assignedClusters = computeCentroids(data, randomCentroids, clusterCount + 1)

        centroidDistancesSum = []
        for centroidIndex, centroid in enumerate(updatedCentroids):
            totalDistance = 0
            for datapointIndex in assignedClusters[centroidIndex]:
                totalDistance += dist(centroid, data[datapointIndex])
            centroidDistancesSum.append(totalDistance)

        xValues.append(clusterCount + 1)
        yValues.append(np.mean(centroidDistancesSum))

    plt.plot(xValues, yValues)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Average Distance per Centroid")
    plt.show()

if __name__ == "__main__":
    random.seed(0)

    # Load dataset
    dataset = 'Dataset/dataset1.csv'
    data   = createDataset(dataset)
    labels = mapDatesToLabels(dataset)
    dataNormed = normalizeDataset(data)

    k = 15
    generateScreePlot(k, dataNormed)

    k = 4
    randomCentroids = random.sample(list(dataNormed), k)
    centroids, clusters = computeCentroids(dataNormed, randomCentroids, k)
    labels = getLabels(clusters, dataNormed, labels)

    for labelIndex, row in enumerate(labels):
        print(f"Cluster {labelIndex + 1} has labels: {Counter(labels[labelIndex]).most_common(4)} ")