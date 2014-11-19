import collections
import numpy
import pylab

def _reportHistData(title, data):

    print title
    pylab.title(title)
    pylab.hist(data, 15)
    pylab.plot()
    pylab.show()
    print '\tAvg:', numpy.average(data)
    print '\tStd:', numpy.std(data)
    print '\tMin:', min(data)
    print '\t25%:', numpy.percentile(data, 25)
    print '\t50%:', numpy.percentile(data, 50)
    print '\t75%:', numpy.percentile(data, 75)
    print '\tMax:', max(data)

"""
Entry point for calculating statistics.

Data = the input data pack.
trim = true if we should not consider attributes a majority of the group has in
common.
"""
def statify(data, trim=False):

    trainingMap = data.trainingMap
    featureMap = data.featureMap
    friendMap = data.friendMap

    # Sample code to find feature intersection within a user's circles.
    print 'Egonet stats:'
    for person in trainingMap:
        print '\tuserid', person, ', who has # friends:', len(friendMap[person]), 'and # circles', len(trainingMap[person])
        print '\tuserid\'s attributes:', data.featureMap[person]
        tooCommonAttrs= {}
        if trim:
            tooCommonAttrs = _findCommonFeaturesPerEgoNet(person, data)
            print '\t\tAttributes common for this egonet:', tooCommonAttrs
        attrs = _findIntersectionFeaturesPerCircle(trainingMap[person], featureMap, 0.25, tooCommonAttrs)
        # Exclude attributes the circle as a whole has in common
        print '\t\t=Intersecting circle attributes:'
        for attr, circle in zip(attrs, trainingMap[person]):
            sortedAttr = sorted(attr.items(), key=lambda x:x[1], reverse=True)
            print '\t\t', len(circle), sortedAttr


    # Calculate useful data
    circleSizes = []
    circleSizesNorm = []
    circleDiameters = []
    circleDiametersNorm = []
    numClusterNormalizeds = []
    avgClusterNormalizeds = []

    for userid in trainingMap:
        numClusterNormalized = 0.0
        avgClusterNormalized = 0.0
        numFriends = len(friendMap[userid])
        for circle in trainingMap[userid]:
            # Size of circle
            circleSizes.append(len(circle))
            circleSizesNorm.append(float(len(circle))/ numFriends)
            # Diameter of the circle )
            diameter = _diameterOf(circle, friendMap)
            circleDiameters.append(diameter)
            circleDiametersNorm.append(float(diameter) / numFriends)
            #
            numClusterNormalized += 1
            #
            avgClusterNormalized += len(circle)


        numClusterNormalized /= numFriends
        avgClusterNormalized /= numFriends
        numClusterNormalizeds.append(numClusterNormalized)
        avgClusterNormalizeds.append(numClusterNormalized)


    # Report data
    _reportHistData('Normalized Number of clusters:', numClusterNormalizeds)
    _reportHistData('Normalized average cluster size:', avgClusterNormalizeds)
    _reportHistData('Normalized circle sizes:', circleSizesNorm)
    _reportHistData('Normalized circle diameters:', circleDiametersNorm)
    _reportHistData('Circle Sizes:', circleSizes)
    _reportHistData('Circle Diameters:', circleDiameters)


def _findCommonFeaturesPerEgoNet(userid, data, percent=0.5):
    commonAttrs = {}

    allAttrs = collections.defaultdict(int)
    for person in data.friendMap[userid]:
        # Count the occurrences per features
        featureMapForPerson = data.featureMap[person]
        for feature in featureMapForPerson:
            allAttrs[feature + featureMapForPerson[feature]] += 1

    # Keep features that have more than a percent people with them.
    for key in allAttrs:
        if allAttrs[key] > percent * len(data.friendMap[userid]):
            commonAttrs[key] = allAttrs[key]

    return commonAttrs


"""
Returns features that people in the circles had in common.
Percent specifies the percent of people that must share the feature in order
for it to be returned.
"""
def _findIntersectionFeaturesPerCircle(circles, featureMap, percent=0.5, exclude={}):
    commonAttrs = []
    for circle in circles:
        commonAttrsForCircle = {}
        allAttrsForCircle = collections.defaultdict(int)

        # Count the occurrences per features
        for person in circle:
            featureMapForPerson = featureMap[person]
            for feature in featureMapForPerson:
                allAttrsForCircle[feature + featureMapForPerson[feature]] += 1

        # Keep features that have more than a percent people with them.
        for key in allAttrsForCircle:
            if (allAttrsForCircle[key] > percent * len(circle)) and (key not in exclude):
                commonAttrsForCircle[key] = allAttrsForCircle[key]

        commonAttrs.append(commonAttrsForCircle)
    return commonAttrs


"""
Uses Floyd-Warshall to compute all pairs shortest path and returns the largest
shortest path over all.
"""
def _diameterOf(circle, friendMap):
    dist = collections.defaultdict(lambda: collections.defaultdict(lambda:
        len(circle)))

    for person in circle:
        dist[person][person] = 0

    for person1 in circle:
        for person2 in circle:
            if person2 in friendMap[person1]:
                dist[person1][person2] = 1

    for i in circle:
        for j in circle:
            for k in circle:
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    maxDist = 0
    for person in circle:
        for person2 in circle:
            if dist[person][person2] > maxDist:
                maxDist = dist[person][person2]

    return maxDist
