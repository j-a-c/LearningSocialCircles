###
# Post-prediction cluster pruning methods. All of these functions should accept
# data, a person, and a list of clusters belonging to the person. The return
# type is a list of clusters returning to the person.
###

"""
Returns the original clusters. Does not make any changes.
"""
def noPrune(data, person, clusters):
    return clusters


def copyBiggest(data, person, clusters):
    lengths = [len(cluster) for cluster in clusters]
    max_length = max(lengths)

    new_clusters = clusters[:]

    for cluster in clusters:
        if len(cluster) == max_length:
            new_clusters.append(clusters)

    return new_clusters
