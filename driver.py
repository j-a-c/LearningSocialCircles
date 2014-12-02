from mcl import mcl
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from clfHelper import attributeAndValue
from stats import statify
from visualize import friendsInCommon
from visualize import originalTopology
from visualize import similarAttributes
from visualize import topologyAndAttributes
from visualize import Visualizer
from userData import Persons
from kmeans import KMeans

import similarityCalculator
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
        self.persons = Persons()

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
    persons = Persons()
    
    for egonetFile in os.listdir(directory):

        currentPerson = egonetFile[:egonetFile.find('.')]
        originalPeople.append(currentPerson)
        persons.addOriginalPerson(currentPerson)
               
        egonetFilePath = os.path.join(directory, egonetFile)

        for line in open(egonetFilePath):
            line = line.strip().split(':')
            currentFriend = line[0]

            friendMap[currentPerson].add(currentFriend)
            friendMap[currentFriend].add(currentPerson)
            persons.getPerson(currentPerson).addFriend(currentFriend)
            persons.getPerson(currentFriend).addFriend(currentPerson)
            
            friends = line[1].strip().split()

            for friend in friends:
                friendMap[currentFriend].add(friend)
                friendMap[friend].add(currentFriend)
                persons.getPerson(currentFriend).addFriend(friend)
                persons.getPerson(friend).addFriend(currentFriend)
            
                friendMap[currentPerson].add(friend)
                friendMap[friend].add(currentPerson)
                persons.getPerson(currentPerson).addFriend(friend)
                persons.getPerson(friend).addFriend(currentPerson)
            
    return friendMap, originalPeople, persons


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
def loadFeatures(filename, persons = None):
    featureMap = collections.defaultdict(dict)
    for line in open(filename):
        parts = line.strip().split()
        currentPerson = parts[0]
        for part in parts[1:]:
            key = part[0:part.rfind(';')]
            value = part[part.rfind(';')+1:]
            featureMap[currentPerson][key] = value
            if persons != None:
                persons.getPerson(currentPerson).addFeature(key, value)
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
    persons = data.persons
    
    sanity = True

    # Check friendMap
    sanity = sanity and (len(friendMap['0']) == 238) and (len(persons.getPerson('0').getFriends()) == 238)
    sanity = sanity and (len(friendMap['850']) == 248) and (len(persons.getPerson('850').getFriends()) == 248)
    # Make sure a person is not included in their own egonet
    for person in friendMap:
        sanity = sanity and (person not in friendMap[person]) and (person not in persons.getPerson(person).getFriends())
    # Check originalPeople
    sanity = sanity and (len(originalPeople) == 110) and (len(persons.getOriginalPersons()) == 110)
    if not sanity:
        print 'Egonets not imported correctly.'
        return sanity

    # Check featureMap
    sanity = sanity and (featureMap['0']['last_name'] == '0') and \
                        (persons.getPerson('0').getFeature('last_name')=='0')
    sanity = sanity and (featureMap['18714']['work;employer;name'] == '12723') and \
                        (persons.getPerson('18714').getFeature('work;employer;name') == '12723')
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

def k_means_clustering(data, show=False):
    attribute_clusters = {}
    attribute_and_friendship_clusters = {}
    k_means = KMeans(data.persons, similarityCalculator.simiarity_according_to_attributes)

    for personID in data.originalPeople:
        clusters = k_means.computeClusters(data.persons.getPerson(personID).getFriends(), 5)
        attribute_clusters[personID] = clusters

    k_means.setSimilarityCalculator(similarityCalculator.simiarity_according_to_attributes_and_friendship)
    for personID in data.originalPeople:
        clusters = k_means.computeClusters(data.persons.getPerson(personID).getFriends(), 5)
        attribute_and_friendship_clusters[personID] = clusters

    if show:
        visualizer = Visualizer()
        for personID in data.persons.getOriginalPersons():
            visualizer.visualizeClusters( attribute_clusters[personID] )
            visualizer.visualizeClusters( attribute_and_friendship_clusters[personID] )

    return attribute_clusters, attribute_and_friendship_clusters


"""
Splits an input list into training and testing sets. percent is the percent of
the input list to be used as the training set.

Return value is the training list and the testing list.
"""
def splitIntoTrainingAndTestSets(trainingPeople, percent=0.5):
    # Shuffle training people and use half for test, half for training.
    print 'Shuffling original training people.'
    random.shuffle(trainingPeople)
    trainingEnd = int(len(trainingPeople) * percent)
    print 'Using', trainingEnd, 'people for training.'
    print 'Using', len(trainingPeople) - trainingEnd, 'people for testing.'
    trainFromTraining = trainingPeople[0:trainingEnd]
    testFromTraining = trainingPeople[trainingEnd:]
    return trainFromTraining, testFromTraining


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
    parser.add_argument('-p', action='store', help='Predict social circles \
            using the given predictor. Supported predictors are \'kmeans\' and \
            \'mcl\'.')
    parser.add_argument('-v', action='store_true', help='Visualize data. By \
            default uses original topology to construct graphs.')
    parser.add_argument('--edge', action='store', help='Select edge function')
    parser.add_argument('--split', action='store_true', help='Split \
            visualizations by circle.')
    parser.add_argument('--save', action='store_true', help='Save output. \
            Graphical output is saved to the folder \'graphs\' in the current \
            directory.')
    parser.add_argument('--show', action='store_true', help='Show output \
            during visualization calculations.')
    args = parser.parse_args()

    # Validate arguments
    EDGE_FUNCS = {
            'top': originalTopology,
            'sim': similarAttributes,
            'tri': friendsInCommon,
            'combo': topologyAndAttributes,
            None: originalTopology
    }
    if args.edge not in EDGE_FUNCS:
        print 'Invalid edge function:', args.edge
        print 'Allowable edge functions:', EDGE_FUNCS
        quit()


    # Input data locations.
    EGONET_DIR = 'egonets'
    TRAINING_DIR = 'training'
    FEATURE_FILE = 'features/features.txt'
    FEATURE_LIST_FILE = 'features/featureList.txt'

    print 'Loading input data.'
    data = Data()

    # Load friend map.
    data.friendMap, data.originalPeople, data.persons = loadEgoNets(EGONET_DIR)

    # Load features.
    data.featureMap = loadFeatures(FEATURE_FILE, data.persons)

    # Load feature list.
    data.featureList = loadFeatureList(FEATURE_LIST_FILE)

    # Load training data.
    data.trainingMap = loadTrainingData(TRAINING_DIR)

    # Sanity checks to make sure we have imported data correctly.
    if not sanityChecks(data):
        print 'Data was not imported in the correct format.'
        exit()

    trainingPeople = []
    for key in data.trainingMap:
        trainingPeople.append(key)

    # Calculate general stats from data.
    if args.s:
        stat_attributes, stat_values = statify(data, args.trim, args.show)

        stat_indices = range(len(stat_attributes))
        train_indices, test_indices = splitIntoTrainingAndTestSets(stat_indices, 0.8)

        stat_training_attrs = [stat_attributes[i] for i in train_indices]
        stat_testing_attrs = [stat_attributes[i] for i in test_indices]

        models = [svm.SVC(), RandomForestClassifier(), LinearRegression(),
                Ridge(), BayesianRidge(), LogisticRegression()]
        model_names = ['SVM', 'Random Forest', 'OLS', 'BayesianRidge',
                'LogisticRegression']
        for value_set in range(len(stat_values)):
            stat_training_values = [stat_values[value_set][i] for i in train_indices]
            stat_testing_values = [stat_values[value_set][i] for i in test_indices]
            print stat_testing_values, '='

            for model, name in zip(models, model_names):

                clf = model
                clf.fit(stat_training_attrs, stat_training_values)
                stat_pred = clf.predict(stat_training_attrs)
                print stat_pred
                diff =[]
                for trueValue, predValue in zip(stat_testing_values, stat_pred):
                    diff.append(abs(trueValue - predValue))
                print 'Errors for model', name, ':', sum(diff), 'on value set', value_set

    # Visualize data
    if args.v:
        visualizer = Visualizer()
        visualizer.visualize(data, EDGE_FUNCS[args.edge], split=args.split,
                save=args.save, show=args.show)

    if args.p:
        # Select prediction method
        if args.p == 'kmeans':
            print 'Using k-means clustering metric.'
            attribute_clusters, attribute_and_friendship_clusters = k_means_clustering(data, args.show)

            real_training_data = 'real_training_data.csv'
            kmeans_attrs = 'kmeans_attrs.csv'
            kmeans_attrs_friends = 'kmeans_attrs_friends.csv'
            kmeans_kaggle_attrs = 'kmeans_kaggle_attrs.csv'
            kmeans_kaggle_attrs_friends = 'kmeans_kaggle_attrs_friends.csv'


            writeSubmission(real_training_data, data.trainingMap)

            # Validation tests
            writeSubmission(kmeans_attrs, {k:attribute_clusters[k] for k in data.trainingMap})
            writeSubmission(kmeans_attrs_friends, {k:attribute_and_friendship_clusters[k] for k in data.trainingMap})

            # Kaggle submissions
            writeSubmission(kmeans_kaggle_attrs, {k:attribute_clusters[k] for k in
                data.originalPeople if k not in data.trainingMap})
            writeSubmission(kmeans_kaggle_attrs_friends,
                    {k:attribute_and_friendship_clusters[k] for k in
                        data.originalPeople if k not in data.trainingMap})

            printMetricCommand(real_training_data, kmeans_attrs)
            printMetricCommand(real_training_data, kmeans_attrs_friends)
            print '\nKaggle submission files:', kmeans_kaggle_attrs, kmeans_kaggle_attrs_friends

        if args.p == 'mcl':
            print 'Using Markov clustering algorithm.'
            print 'Using edge function:', args.edge
            # Markov Cluster Algorithm
            mclCirclMap = {}
            counter = 1
            for currentPerson in trainingPeople:
                # Report progress
                print counter, '/', len(trainingPeople)
                counter += 1

                # Perform actual MCL calculation
                mclCircles = None
                if args.edge:
                    # Build weights
                    weights = collections.defaultdict(dict)
                    for friend1 in data.friendMap[currentPerson]:
                        weight = EDGE_FUNCS[args.edge](data, friend1, currentPerson)
                        weights[friend1][currentPerson] = weight
                        weights[currentPerson][friend1] = weight

                        for friend2 in data.friendMap[currentPerson]:
                            weight = EDGE_FUNCS[args.edge](data, friend1, friend2)
                            weights[friend1][friend2] = weight
                            weights[friend2][friend1] = weight



                    mclCircles = mcl(currentPerson, data, weights)
                else:
                    mclCircles = mcl(currentPerson, data)
                mclTotalPeople = 0
                actualTotalPeople = 0
                for circle in mclCircles:
                    mclTotalPeople += len(circle)
                for circle in data.trainingMap[currentPerson]:
                    actualTotalPeople += len(circle)

                mclCirclMap[currentPerson] = mclCircles
                print 'Num MCL circles:', len(mclCircles), 'Actual:', len(data.trainingMap[currentPerson])
                print '\tMCL in:', mclTotalPeople, 'Actual in:', actualTotalPeople

            mcl_training = 'mcl_training.csv'
            real_training = 'real_training_data.csv'

            writeSubmission(mcl_training, mclCirclMap)
            writeSubmission(real_training, data.trainingMap)
            printMetricCommand(real_training, mcl_training)

        # Default is 'All friends in one cricle' metric'.
        else:
            print 'Using one-circle metric.'
            # All friends in one circle submission.
            # This is just a test to check our input against sample_submission.csv.
            # To run: 'socialCircles_metric.py sample_submission.csv one_circle.csv'
            oneCircleMap = {}
            for person in data.originalPeople:
                if not person in data.trainingMap:
                    oneCircleMap[person] = data.friendMap[person]
            writeSubmission('one_circle.csv', oneCircleMap, True)
            printMetricCommand('sample_submission.csv', 'one_circle.csv')



        """
        TODO Clean up lines below here...
        """

        quit()




        # SVM w/ Markov Cluster
        svmMclCircleMap = {}

        trainFromTraining, testFromTraining = splitIntoTrainingAndTestSets(trainingPeople)

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
