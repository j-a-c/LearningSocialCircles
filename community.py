import igraph
from sets import Set

def name(sourceName, targetName):
    if sourceName < targetName:
        return str(sourceName) + '-' + str(targetName)
    else:
        return str(targetName) + '-' + str(sourceName)

def extract_clusters(clusters, reverseIdMap):
    new_clusters = []
    for i in range(len(clusters)):
        next_cluster = [reverseIdMap[j] for j in clusters[i]]
        new_clusters.append(next_cluster)
    return new_clusters

def community_using_igraph(data, origPerson, edgeFunc, pruneFunc):
    friendMap = data.friendMap

    # We will map ids to indices in the Graph.
    idMap = {}
    reverseIdMap = {}

    added_edges = Set()

    numFriends = len(friendMap[origPerson])
    idMap[origPerson] = numFriends - 1

    # Create person to index mappings
    currentId = 0
    for friend in friendMap[origPerson]:
        idMap[friend] = currentId
        reverseIdMap[currentId] = friend
        currentId += 1

    g = igraph.Graph()
    g.es['weight'] = 1.0

    g.add_vertices(numFriends)

    for source in idMap:
        for target in idMap:
            # Do not add self edges
            if source != target:
                sourceName = idMap[source]
                targetName = idMap[target]
                edgeName = name(sourceName, targetName)
                # Do not add the same edge twice
                weight = edgeFunc(data, source, target)
                if not edgeName in added_edges and weight:
                    g.add_edge(sourceName, targetName)
                    added_edges.add(edgeName)
                    g[sourceName, targetName] = weight

    clusters = []


    # vd_betweenness = g.community_edge_betweenness(directed=False)
    #TODO print vd_betweenness#.as_clustering(5)

    # vd_fastgreedy = g.community_fastgreedy()
    #TODO print vd_fastgreedy

    try:
        # edge_weights, vertex_weights
        clusters = g.community_infomap(edge_weights='weight')
    except:
        clusters = []
    infomap_clusters = extract_clusters(clusters, reverseIdMap)
    infomap_clusters = pruneFunc(data, origPerson, infomap_clusters)

    """
    try:
        # weights
        #clusters = g.community_leading_eigenvector(weights='weight')
        None
    except:
        clusters = []
    eigen_clusters = extract_clusters(clusters, reverseIdMap)

    try:
        # weights
        #clusters = g.community_label_propagation(weights='weights')
        None
    except:
        clusters = []
    label_clusters = extract_clusters(clusters, reverseIdMap)

    try:
        # weights
        #clusters = g.community_multilevel(weights='weight')
        None
    except:
        clusters = []
    multi_clusters = extract_clusters(clusters, reverseIdMap)

    try:
        None
        #clusters = g.community_spinglass(weights='weight')
    except:
        clusters = []
    spin_clusters = extract_clusters(clusters, reverseIdMap)

    #TODO print g.community_walktrap()

    return infomap_clusters, eigen_clusters, label_clusters, multi_clusters, spin_clusters
    """
    return infomap_clusters, infomap_clusters, infomap_clusters, infomap_clusters, infomap_clusters

