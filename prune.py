###
# Post-prediction cluster pruning methods. All of these functions should accept
# data, a person, and a list of clusters belonging to the person. The return
# type is a list of clusters returning to the person.
###

def _removeOriginalPersonFromClusters(original, clusters):
    new_clusters = []
    for cluster in clusters:
        new_clusters.append([i for i in cluster if i != original])
    return new_clusters

"""
Returns the original clusters. Does not make any changes.
"""
def noPrune(data, person, clusters):
    clusters = _removeOriginalPersonFromClusters(person, clusters)
    return clusters

def removeClustersWithNotManyPeople(data, person, clusters):
    new_clusters = []
    for cluster in clusters:
        if len(clusters) > 3:
            new_clusters.append(cluster)
    return new_clusters

def copyBiggest(data, person, clusters):
    lengths = [len(cluster) for cluster in clusters]
    max_length = max(lengths)

    new_clusters = clusters[:]

    for cluster in clusters:
        if len(cluster) == max_length:
            new_clusters.append(cluster)

    return new_clusters
