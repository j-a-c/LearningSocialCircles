import collections

"""
Returns a (featuresList, trueValue) pair for person1 and person2.
origPerson is the main friend who person1 and person2 are friends with.

Feature vector (We are comparing person 1 and person 2):
    abs(Total friends for person 1 - Total friends for person 2)
    Percent of total friends in common
    Number of features in common
    #Common features at depth
    #Common features at depth with origPerson
    Exact 3 depth (education, work)
"""
def attributeAndValue(origPerson, person1, person2, data):
    friendMap = data.friendMap
    featureMap = data.featureMap

    # Initial attributes and value
    nextAttr = []
    # abs(Total friends for person 1 - Total friends for person 2)
    numFriendDiff = len(friendMap[person1]) - len(friendMap[person2])
    numFriendDiff = abs(numFriendDiff)
    #nextAttr.append(numFriendDiff)
    # Total friends in common
    person1sFriends = friendMap[person1]
    numFriendsInCommon = 0.0
    for person2Friend in friendMap[person2]:
        if person2Friend in person1sFriends:
            numFriendsInCommon += 1
    numFriendsInCommon /= len(friendMap[origPerson])
    nextAttr.append(numFriendsInCommon)
    # Number of features in common 
    numFeaturesInCommon = 0
    commonAtDepth = [0,0,0,0]
    commonAtDepthWithOrig = [0,0,0,0]
    exact3Depth = collections.defaultdict(int)
    exact2Depth = collections.defaultdict(int)
    exact3DepthKeys = ['edu', 'work', 'home', 'gender', 'loc', 'last', 'lang']
    for key in featureMap[person1]:
        if key in featureMap[person2]:
            if featureMap[person1][key] == featureMap[person2][key]:
                commonAtDepth[len(key.split(';')) - 1] += 1
                numFeaturesInCommon += 1

                for exact3DepthKey in exact3DepthKeys:
                    if key.startswith(exact3DepthKey):
                        #exact3Depth[exact3DepthKey] = max(commonDepth, exact3Depth[exact3DepthKey])
                        exact2Depth[exact3DepthKey] += 1


                if (key in featureMap[origPerson]) and featureMap[origPerson][key] == featureMap[person1][key]:
                    commonDepth = len(key.split(';')) - 1
                    commonAtDepthWithOrig[commonDepth] += 1

                    for exact3DepthKey in exact3DepthKeys:
                        if key.startswith(exact3DepthKey):
                            #exact3Depth[exact3DepthKey] = max(commonDepth, exact3Depth[exact3DepthKey])
                            exact3Depth[exact3DepthKey] += 1
    numFeaturesInCommon = (2.0 * numFeaturesInCommon) / (len(featureMap[person1]) + len(featureMap[person2]))

    nextAttr.append(numFeaturesInCommon)

    # Common features at depth
    for depth in commonAtDepth:
        nextAttr.append(depth)
    #for depth in commonAtDepthWithOrig:
    #    nextAttr.append(depth)
    #for exact3DepthKey in exact3DepthKeys:
    #    #nextAttr.append(exact3Depth[exact3DepthKey])
    #    nextAttr.append(exact2Depth[exact3DepthKey])



    nextValue = 0
    # Calculate actual value
    for circle in data.trainingMap[origPerson]:
        if (person1 in circle) and (person2 in circle):
            nextValue = 1


    return nextAttr, nextValue
