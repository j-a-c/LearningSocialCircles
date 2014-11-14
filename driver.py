from mcl import mcl

import circleStats
import collections
import numpy
import os
import pylab

"""
Each file in this directory contains the ego-network of a
single Facebook user, i.e., a list of connections between their
friends. Each file userId.egonet contains lines of the form

UserId: Friends
1: 4 6 12 2 208
2: 5 3 17 90 7

These are node-adjacency lists indicating that User 1 is friends with
Friend 4, Friend 6, etc. Edges are undirected in Facebook. It can be
assumed that user (the owner of the egonet) is friends with all ids
in this file.

File names are of the form userid.egonetFile

Returns a map (userids -> set of friends) and a list of the original friend ids
to consider.
"""
def loadEgoNets(directory):
    friendMap = collections.defaultdict(set)
    originalPeople = []

    for egonetFile in os.listdir(directory):

        currentPerson = egonetFile[:egonetFile.find('.')]
        originalPeople.append(currentPerson)

        egonetFilePath = os.path.join(directory, egonetFile)

        for line in open(egonetFilePath):
            line = line.strip().split(':')
            currentFriend = line[0]

            friendMap[currentPerson].add(currentFriend)
            friendMap[currentFriend].add(currentPerson)

            friends = line[1].strip().split()

            for friend in friends:
                friendMap[currentFriend].add(friend)
                friendMap[friend].add(currentFriend)

                friendMap[currentPerson].add(friend)
                friendMap[friend].add(currentPerson)
    return friendMap, originalPeople


"""
Input is a directory containing files with file names UserId.circles.
Each file contains lines of the form:

circleID: friend1 friend2 friend3 ...

describing the circles for UserId.

Returns a map from UserId -> a list of circles
"""
def loadTrainingData(directory):
    trainingMap = collections.defaultdict(list)

    for trainingFile in os.listdir(directory):
        currentPerson = trainingFile[:trainingFile.find('.')]

        trainingFilePath = os.path.join(directory, trainingFile)
        for line in open(trainingFilePath):
            parts = line.strip().split()
            trainingMap[currentPerson].append(parts[1:])

    return trainingMap


"""

"""
def calculateSimilarityDepth(feature1):
    return len(feature1.split(';'))


"""
Input is a file name whose content is of the form:

0 first_name;435 hometown;name;567 work;employer;name;23

per line. The first number is the UserId and the rest of the line are
feature/value mappings.

Returns a map mappings UserId to a map of features.
"""
def loadFeatures(filename):
    featureMap = collections.defaultdict(dict)
    for line in open(filename):
        parts = line.strip().split()
        currentPerson = parts[0]
        for part in parts[1:]:
            key = part[0:part.rfind(';')]
            value = part[part.rfind(';')+1:]
            featureMap[currentPerson][key] = value
    return featureMap


"""
Input is a file name whose content is a list of all possible features, with one
feature per line.

Ouput is a list containing all the features.
"""
def loadFeatureList(filename):
    featureList = []
    for line in open(filename):
        featureList.append(line.strip())
    return featureList


"""
Simple tests to make sure the input is in the correct format.
"""
def sanityChecks(friendMap, originalPeople, featureMap, featureList,
        trainingMap):
    sanity = True

    # Check friendMap
    sanity = sanity and (len(friendMap['0']) == 238)
    sanity = sanity and (len(friendMap['850']) == 248)
    # Check originalPeople
    sanity = sanity and (len(originalPeople) == 110)
    if not sanity:
        print 'Egonets not imported correctly.'
        return sanity

    # Check featureMap
    sanity = sanity and (featureMap['0']['last_name'] == '0')
    sanity = sanity and (featureMap['18714']['work;employer;name'] == '12723')
    if not sanity:
        print 'Features not imported correctly.'
        return sanity

    # Check length of feature list
    sanity = sanity and (len(featureList) == 57)
    if not sanity:
        print 'Feature list imported incorrectly.'
        return sanity

    # Check trainingMap
    sanity = sanity and (len(trainingMap) == 60)
    sanity = sanity and (len(trainingMap['2255']) == 3)
    sanity = sanity and (len(trainingMap['2255'][0]) == 51)
    sanity = sanity and (len(trainingMap['2255'][1]) == 6)
    sanity = sanity and (len(trainingMap['2255'][2]) == 93)
    if not sanity:
        print 'Training data not imported correctly.'
        return sanity

    return sanity


"""
Writes output in the specified format:

userId ((space separated list of friend ids in circle);)*
"""
def writeSubmission(filename, circleMap, test=False):
    f = open(filename, 'w+')

    f.write('UserId,Predicted\n')

    for person, circles in circleMap.iteritems():

        line = person + ','

        if not test:
            for circle in circles:
                for friend in circle:
                    line += friend + ' '
                line += ';'
        else:
            for friend in circles:
                line += friend + ' '
            line += ';'


        line += '\n'
        f.write(line)

    f.close()


"""
Not all friends have to be in a circle.
Circles may be disjoint, overlap, or hierarchically nested.

Strategies:
    1. Find which features circles commonly share.
    2. Find connected components within friend graph.
"""
if __name__ == '__main__':
    EGONET_DIR = 'egonets'
    TRAINING_DIR = 'Training'
    FEATURE_FILE = 'features.txt'
    FEATURE_LIST_FILE = 'featureList.txt'

    # Load friend map.
    friendMap, originalPeople = loadEgoNets(EGONET_DIR)

    # Load features.
    featureMap = loadFeatures(FEATURE_FILE)

    # Load feature list.
    featureList = loadFeatureList(FEATURE_LIST_FILE)

    # Load training data.
    trainingMap = loadTrainingData(TRAINING_DIR)

    # Sanity checks to make sure we have imported data correctly.
    sanity = sanityChecks(friendMap, originalPeople, featureMap, featureList,
            trainingMap)
    if not sanity:
        print 'Data was not imported in the correct format.'
        exit()

    """
    # All friends in one circle submission.
    # This is just a test to check our input against sample_submission.csv.
    # To run: 'socialCircles_metric.py sample_submission.csv one_circle.csv'
    oneCircleMap = {}
    for person in originalPeople:
        if not person in trainingMap:
            oneCircleMap[person] = friendMap[person]
    writeSubmission('one_circle.csv', oneCircleMap, True)
    """

    """
    # Sample code to find feature intersection within a user's circles.
    print 'Intersecting attrs for user 611'
    attrs = cicleStats.findIntersectionFeaturesPerCircle(trainingMap['611'], featureMap, 0.25)
    for attr, circle in zip(attrs, trainingMap['611']):
        print len(circle), attr
    """

    """
    # Calculate useful data
    circleSizes = []
    circleDiameters = []

    for userid in trainingMap:
        for circle in trainingMap[userid]:
            # Size of circle
            circleSizes.append(len(circle))
            # Diameter of the circle )
            circleDiameters.append(circleStats.diameterOf(circle, friendMap))


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

    trainingPeople = []
    for key in trainingMap:
        trainingPeople.append(key)


    # Markov Cluster Algorithm
    mclCirclMap = {}
    counter = 1
    for currentPerson in trainingPeople:
        # Report progress
        print counter, '/', len(trainingPeople)
        counter += 1

        # Perform actual MCL calculation
        mclCircles = mcl(currentPerson, friendMap)
        mclTotalPeople = 0
        actualTotalPeople = 0
        for circle in mclCircles:
            mclTotalPeople += len(circle)
        for circle in trainingMap[currentPerson]:
            actualTotalPeople += len(circle)

        mclCirclMap[currentPerson] = mclCircles
        #print 'Num MCL circles:', len(mclCircles), 'Actual:', len(trainingMap[currentPerson])
        #print 'MCL in:', mclTotalPeople, 'Actual in:', actualTotalPeople
    writeSubmission('mcl_circle.csv', mclCirclMap)
    writeSubmission('real_circle.csv', trainingMap)
