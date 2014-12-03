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



def community_using_igraph(data, origPerson, edgeFunc):
    friendMap = data.friendMap

    # Create adjacency matrix. We will map ids to indices in the matrix.
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

    g.add_vertices(numFriends)
    print origPerson
    print numFriends, currentId
    for source in idMap:
        for target in friendMap[source]:
            # Do not add self edges
            if source != target:
                sourceName = idMap[source]
                targetName = idMap[target]
                edgeName = name(sourceName, targetName)
                # Do not add the same edge twice
                if not edgeName in added_edges and edgeFunc(data, source, target):
                    g.add_edge(sourceName, targetName)
                    added_edges.add(edgeName)


    #g.add_edges([[0,1], [1,2], [2,0], [2,3], [3,4], [4,5]])


    vd_betweenness = g.community_edge_betweenness(directed=False)
    print vd_betweenness#.as_clustering(5)

    vd_fastgreedy = g.community_fastgreedy()
    print vd_fastgreedy

    print '= info map'
    clusters = g.community_infomap()
    infomap_clusters = extract_clusters(clusters, reverseIdMap)

    print '= lead eigen'
    clusters = g.community_leading_eigenvector()
    eigen_clusters = extract_clusters(clusters, reverseIdMap)

    print '= label prop'
    print g.community_label_propagation()

    print '= multilevel'
    print g.community_multilevel()

    print '= spinglass'
    print g.community_spinglass()

    print g.community_walktrap()

    print 'Actual # clusters:', len(data.trainingMap[origPerson])
    print 'Actual clusters:'
    print data.trainingMap[origPerson]

    return infomap_clusters, eigen_clusters
