from sets import Set

###
# Acceptable weight functions.
# Must return a value that is logically true (number > 0) if an edge should
# exist between the two people, and None if no edge should exists.
###

class Igraphh_WeightCalculator:
    def __init__(self, featureWeightMap):
        self._featureWeightMap = featureWeightMap
        
    """
    Returns true if an edge exists between the two people in the orginal topology.
    """
    def originalTopology(self, data, person1, person2):
        if person1 in data.friendMap[person2]:
            return 1.0
        return None
    
    
    def friendsInCommon(self, data, person1, person2, threshold=3):
        numFriendsInCommon = 0
        for friend in data.friendMap[person1]:
            if friend in data.friendMap[person2]:
                numFriendsInCommon += 1
        if numFriendsInCommon > threshold:
            return 1.0
        else:
            return None
    
    
    """
    Returns true if the two people have more than 'threshold' attributes in common.
    Does not include the amount of friends in common.
    """
    def similarAttributes(self, data, person1, person2, threshold=3):
        numberAttributesInCommon = 0
        for key in data.featureMap[person1]:
            person1Value = data.featureMap[person1][key]
            person2Value = None
            if key in data.featureMap[person2]:
                person2Value = data.featureMap[person2][key]
            if person1Value == person2Value:
                numberAttributesInCommon += 1
        if numberAttributesInCommon > threshold:
            return 1.0
        return None
    
    def topologyAndAttributes(self, data, person1, person2):
        if self.similarAttributes(data, person1, person2) or self.friendsInCommon(data,
                person1, person2):
            return 1.0
        else:
            return None

    
    
       
    def weightedAttrubuites(self, data, person1, person2):
        weightAttributes = 0
        
        for person1FeatureKey in data.featureMap[person1]:
            person1FeatureValue = data.featureMap[person1][person1FeatureKey]
            person2FeatureValue = None
            if person1FeatureKey in data.featureMap[person2]:
                person2FeatureValue = data.featureMap[person2][person1FeatureKey]
            if person1FeatureValue == person2FeatureValue:
                weightAttributes = weightAttributes + self._featureWeightMap[person1FeatureKey]
        
        normalizedWeightAttributes = (weightAttributes/sum(self._featureWeightMap.itervalues()))
        
        if normalizedWeightAttributes == 0:
            return None        
        else:
            return normalizedWeightAttributes 
    
    def originalTopologyAndWeightedAttrubuites(self, data, person1, person2):
        weightAttributes = self.weightedAttrubuites(data, person1, person2)
        weightOriginalTopology = self.originalTopology(data, person1, person2)
        
        if weightAttributes == None:
            return weightOriginalTopology
        if weightOriginalTopology == None:
            return weightAttributes
        
        return weightAttributes + weightOriginalTopology
    

def attributeIntersection(data, person1, person2):
    profile_attrs1 = Set()
    profile_attrs2 = Set()

    for feature in data.featureMap[person1]:
        profile_attrs1.add(data.featureMap[person1][feature])
    for feature in data.featureMap[person2]:
        profile_attrs2.add(data.featureMap[person2][feature])

    intersection = len(profile_attrs1.intersection(profile_attrs2))
    union = len(profile_attrs1.union(profile_attrs2))
    if intersection < 5:
        return None
    else:
        return (1.0 * intersection) / union


def originalTopologyAndAttributeIntersection(data, person1, person2):
    weight1 = originalTopology(data, person1, person2)
    weight2 = attributeIntersection(data, person1, person2)

    if weight1 == None:
        return weight2
    if weight2 == None:
        return weight1

    return weight1 + weight2
