from mcl import mcl
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from clfHelper import attributeAndValue
from stats import statify
from visualize import originalTopology
from visualize import similarAttributes
from visualize import visualize

import argparse
import collections
import os
import random

"""
"""
class Data:
    def __init__(self):
        self.friendMap = None
        self.originalPeople = None
        self.featureMap = None
        self.featureList = None
        self.trainingMap = None


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
def sanityChecks(data):
    friendMap = data.friendMap
    originalPeople = data.originalPeople
    featureMap = data.featureMap
    featureList = data.featureList
    trainingMap = data.trainingMap

    sanity = True

    # Check friendMap
    sanity = sanity and (len(friendMap['0']) == 238)
    sanity = sanity and (len(friendMap['850']) == 248)
    # Make sure a person is not included in their own egonet
    for person in friendMap:
        sanity = sanity and (person not in friendMap[person])
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

def printMetricCommand(realOutput, testOutput):
    print '\nEvaluate using:'
    print 'python socialCircles_metric.py', realOutput, testOutput


"""
Not all friends have to be in a circle.
Circles may be disjoint, overlap, or hierarchically nested.

Strategies:
    1. Find which features circles commonly share.
    2. Find connected components within friend graph.
"""
if __name__ == '__main__':
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Process social circle data.')
    parser.add_argument('-s', action='store_true', help='Compute statistics.')
    parser.add_argument('--trim', action='store_true', help='Trim common data.')
    parser.add_argument('-p', action='store_true', help='Predict social circles.')
    parser.add_argument('-v', action='store_true', help='Visualize data. By \
            default uses original topology to construct graphs.')
    parser.add_argument('--edge', action='store', help='Select edge function')
    args = parser.parse_args()

    # Validate arguments
    EDGE_FUNCS = ['top', 'sim']
    if args.edge not in EDGE_FUNCS:
        print 'Invalid edge function:', args.edge
        print 'Allowable edge functions:', EDGE_FUNCS
        quit()


    # Input data locations.
    EGONET_DIR = 'egonets'
    TRAINING_DIR = 'training'
    FEATURE_FILE = 'features/features.txt'
    FEATURE_LIST_FILE = 'features/featureList.txt'

    data = Data()

    # Load friend map.
    data.friendMap, data.originalPeople = loadEgoNets(EGONET_DIR)

    # Load features.
    data.featureMap = loadFeatures(FEATURE_FILE)

    # Load feature list.
    data.featureList = loadFeatureList(FEATURE_LIST_FILE)

    # Load training data.
    data.trainingMap = loadTrainingData(TRAINING_DIR)

    # Sanity checks to make sure we have imported data correctly.
    if not sanityChecks(data):
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
    printMetricCommand('sample_submission.csv', 'one_circle.csv')
    """

    # Calculate general stats from data.
    if args.s and args.trim:
        statify(data, True)
    elif args.s:
        statify(data)

    trainingPeople = []
    for key in data.trainingMap:
        trainingPeople.append(key)


    """
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
    """

    # Visualize data
    if args.v and args.edge == 'top':
        visualize(data, originalTopology)
    elif args.v and args.edge == 'sim':
        visualize(data, similarAttributes)
    elif args.v:
        visualize(data)

    if args.p:
        # SVM w/ Markov Cluster
        svmMclCircleMap = {}

        # Shuffle training people and use half for test, half for training.
        print 'Shuffling original training people.'
        random.shuffle(trainingPeople)
        trainingEnd = int(len(trainingPeople) * 0.5)
        print 'Using', trainingEnd, 'people for training.'
        print 'Using', len(trainingPeople) - trainingEnd, 'people for testing.'
        trainFromTraining = trainingPeople[0:trainingEnd]
        testFromTraining = trainingPeople[trainingEnd:]

        # Attempt to train SVM to detect if two people are in same circle.
        SVMTrainAttrs = []
        SVMTrainValues = []
        # Create training data for SVM.
        print 'Creating training data for SVM.'
        for currentPerson in trainFromTraining:

            # We want to include each pair of friends only once.
            friendList = []
            for friend in data.friendMap[currentPerson]:
                # Safety check to avoid self loops
                if friend != currentPerson:
                    friendList.append(friend)

            for firstIndex in range(len(friendList)):
                for secondIndex in range(firstIndex+1, len(friendList)):
                    person1 = friendList[firstIndex]
                    person2 = friendList[secondIndex]

                    nextAttr, nextValue = attributeAndValue(currentPerson, person1, person2, data)

                    SVMTrainAttrs.append(nextAttr)
                    SVMTrainValues.append(nextValue)
        # Train SVM
        print 'Total training values:', len(SVMTrainValues)
        #clf = svm.SVC()
        clf = RandomForestClassifier()

        """
        # Use this code if it is too slow to train model with lot of examples.
        # Keep a subset of training data
        NUM_TRAIN = 500
        print 'Selecting', NUM_TRAIN, 'values for training.'
        selected = [random.randint(0, len(SVMTrainAttrs)) for i in range(NUM_TRAIN)]
        SVMTrainAttrs = [SVMTrainAttrs[i] for i in selected]
        SVMTrainValues = [SVMTrainValues[i] for i in selected]
        """
        #
        print 'Training SVM.'
        clf.fit(SVMTrainAttrs, SVMTrainValues)

        # Test SVM.
        print 'Testing SVM.'
        testSVMAttrs = []
        testSVMTrueValues = []
        friendPairs = []
        weights = collections.defaultdict(dict)
        for currentPerson in testFromTraining:

            # We want to include each pair of friends only once.
            friendList = []
            for friend in data.friendMap[currentPerson]:
                # Safety check to avoid self loops
                if friend != currentPerson:
                    friendList.append(friend)

            for firstIndex in range(len(friendList)):
                for secondIndex in range(firstIndex+1, len(friendList)):
                    person1 = friendList[firstIndex]
                    person2 = friendList[secondIndex]

                    friendPairs.append((person1, person2))
                    nextAttr, nextValue = attributeAndValue(currentPerson, person1, person2, data)
                    testSVMAttrs.append(nextAttr)
                    testSVMTrueValues.append(nextValue)

        testSVMPredValues = clf.predict(testSVMAttrs)
        numTestsInInCorrect = 0
        numTestsInWrong = 0
        numTestsInInTotal = 0
        numTestOutWrong = 0
        numTestOutTotal = 0
        numTestOutCorrect = 0
        for trueValue, predValue, pair in zip(testSVMTrueValues, testSVMPredValues, friendPairs):
            weights[pair[0]][pair[1]] = predValue
            weights[pair[1]][pair[0]] = predValue
            if trueValue == predValue:
                if trueValue == 1:
                    numTestsInInCorrect += 1
                else:
                    numTestOutCorrect += 1
            # trueValue != predValue
            else:
                if trueValue == 0:
                    numTestOutWrong += 1
                else:
                    numTestsInWrong += 1


            if trueValue == 1:
                numTestsInInTotal += 1
            else:
                numTestOutTotal += 1

        totalCorrect =  + numTestsInInCorrect + numTestOutCorrect
        print 'Total Correct:', totalCorrect, '/', len(testSVMPredValues)
        print '\t%:', (1.0 * totalCorrect) / len(testSVMPredValues)
        print 'True In Circle:', numTestsInInCorrect, '/', numTestsInInTotal, 'Want this to be high!)'
        print '\t%:', (1.0 * numTestsInInCorrect) / numTestsInInTotal
        print 'False In Circle:', numTestOutWrong, '/', numTestOutTotal, '(Want this to be low!)'
        print '\t%:', (1.0 * numTestOutWrong) / numTestOutTotal
        #print 'True Out Circle:', numTestOutCorrect, '/', numTestOutTotal
        #print '\t%:', (1.0 * numTestOutCorrect) / numTestOutTotal
        #print 'False Out Circle:', numTestsInWrong, '/', numTestsInInTotal
        #print '\t%:', (1.0 * numTestsInWrong) / numTestsInInTotal


        # Use SVM to construct SVM matrix
        print 'Predicting circles.'
        testFromTrainingMap = {}
        counter = 1
        for currentPerson in testFromTraining:
            # Report progress
            print counter, '/', len(testFromTraining)
            counter += 1

            # Perform actual MCL calculation
            mclCircles = mcl(currentPerson, data, weights)
            mclTotalPeople = 0
            actualTotalPeople = 0
            for circle in mclCircles:
                mclTotalPeople += len(circle)
            for circle in data.trainingMap[currentPerson]:
                actualTotalPeople += len(circle)

            svmMclCircleMap[currentPerson] = mclCircles
            testFromTrainingMap[currentPerson] = data.trainingMap[currentPerson]
            print 'Num MCL circles:', len(mclCircles), 'Actual:', len(data.trainingMap[currentPerson])
            print 'MCL in:', mclTotalPeople, 'Actual in:', actualTotalPeople
        TEST_OUTPUT = 'svm_mcl_circle.csv'
        REAL_OUTPUT = 'real_circle.csv'
        writeSubmission(TEST_OUTPUT, svmMclCircleMap)
        writeSubmission(REAL_OUTPUT, testFromTrainingMap)
        printMetricCommand(REAL_OUTPUT, TEST_OUTPUT)
