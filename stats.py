import collections
import numpy
import pylab

"""
Entry point for calculating statistics.
"""
def statify(data):

    trainingMap = data.trainingMap
    featureMap = data.featureMap
    friendMap = data.friendMap

    # Sample code to find feature intersection within a user's circles.
    print 'Intersecting attrs:'
    for person in trainingMap:
        print person
        attrs = _findIntersectionFeaturesPerCircle(trainingMap[person], featureMap, 0.25)
        for attr, circle in zip(attrs, trainingMap[person]):
            sortedAttr = sorted(attr.items(), key=lambda x:x[1], reverse=True)
            print '\t', len(circle), sortedAttr


    # Calculate useful data
    circleSizes = []
    circleDiameters = []

    for userid in trainingMap:
        for circle in trainingMap[userid]:
            # Size of circle
            circleSizes.append(len(circle))
            # Diameter of the circle )
            circleDiameters.append(_diameterOf(circle, friendMap))


    # Report data

    # Size of circle
    print 'Circle Size Data:'
    pylab.title('Histogram of Circle Sizes')
    pylab.hist(circleSizes, 15)
    pylab.plot()
    pylab.show()
    print '\tAvg:', numpy.average(circleSizes)
    print '\tStd:', numpy.std(circleSizes)
    print '\tMin:', min(circleSizes)
    print '\t25%:', numpy.percentile(circleSizes, 25)
    print '\t50%:', numpy.percentile(circleSizes, 50)
    print '\t75%:', numpy.percentile(circleSizes, 75)
    print '\tMax:', max(circleSizes)


    # Diameter of circles
    print 'Circle Diameter Data:'
    pylab.title('Histogram of Circle Diameters')
    pylab.hist(circleDiameters, 15)
    pylab.plot()
    pylab.show()
    print '\tAvg:', numpy.average(circleDiameters)
    print '\tStd:', numpy.std(circleDiameters)
    print '\tMin:', min(circleDiameters)
    print '\t25%:', numpy.percentile(circleDiameters, 25)
    print '\t50%:', numpy.percentile(circleDiameters, 50)
    print '\t75%:', numpy.percentile(circleDiameters, 75)
    print '\tMax:', max(circleDiameters)


"""
Returns features that people in the circles had in common.
Percent specifies the percent of people that must share the feature in order
for it to be returned.
"""
def _findIntersectionFeaturesPerCircle(circles, featureMap, percent=0.5):
    commonAttrs = []
    for circle in circles:
        commonAttrsForCircle = {}
        allAttrsForCircle = collections.defaultdict(int)

        # Count the occurrences per features
        for person in circle:
            featureMapForPerson = featureMap[person]
            for feature in featureMapForPerson:
                allAttrsForCircle[feature + featureMapForPerson[feature]] += 1

        # Keep features that have more than 2 people with them.
        for key in allAttrsForCircle:
            if allAttrsForCircle[key] > percent * len(circle):
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
