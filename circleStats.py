import collections

"""
Returns features that people in the circles had in common.
Percent specifies the percent of people that must share the feature in order
for it to be returned.
"""
def findIntersectionFeaturesPerCircle(circles, featureMap, percent=0.5):
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
def diameterOf(circle, friendMap):
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
