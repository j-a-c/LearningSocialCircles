
class SimilarityCalculator(object):
    def __init__(self, featureWeightMap):
        self._featureWeightMap = featureWeightMap 
       
    def similarity_weighted_attributes_friendship( self, persons, friend1ID, friend2ID ):
        FRIENDSHIP_WEIGHT = 2.5
        similarity = 0
        
        friend1 = persons.getPerson(friend1ID)
        friend2 = persons.getPerson(friend2ID)
        
        for key in friend1.getFeatures():
            friend1Value = friend1.getFeature(key)
            friend2Value = friend2.getFeature(key)
            if friend1Value == friend2Value:
                similarity += self._featureWeightMap[key]
                
        #normalize
        similarity = (similarity/sum(self._featureWeightMap.itervalues()))*10
        
        #consider friendship 
        if friend1.isFriend(friend2ID):
            similarity += FRIENDSHIP_WEIGHT        
        
        return similarity
    
    def simiarity_according_to_attributes( self, persons, friend1ID, friend2ID ):
        number_of_common_attributes = 0
        
        friend1 = persons.getPerson(friend1ID)
        friend2 = persons.getPerson(friend2ID)
        
        for key in friend1.getFeatures():
            friend1Value = friend1.getFeature(key)
            friend2Value = friend2.getFeature(key)
            if friend1Value == friend2Value:
                number_of_common_attributes += 1
                        
        return (number_of_common_attributes + 1)
    
    
    def simiarity_according_to_attributes_and_friendship( self, persons, friend1ID, friend2ID ):
        FRIENDSHIP_WEIGHT = 10
        number_of_common_attributes = 0
        
        friend1 = persons.getPerson(friend1ID)
        friend2 = persons.getPerson(friend2ID)
        
        for key in friend1.getFeatures():
            friend1Value = friend1.getFeature(key)
            friend2Value = friend2.getFeature(key)
            if friend1Value == friend2Value:
                number_of_common_attributes += 1
        
        if friend1.isFriend(friend2ID):
            return (number_of_common_attributes + 1 + FRIENDSHIP_WEIGHT)        
        
        return (number_of_common_attributes + 1)

