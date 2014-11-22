
FRIENDSHIP_WEIGHT = 20

def simiarity_according_to_attributes( persons, friend1ID, friend2ID ):
    number_of_common_attributes = 0
    
    friend1 = persons.getPerson(friend1ID)
    friend2 = persons.getPerson(friend2ID)
    
    for key in friend1.getFeatures():
        friend1Value = friend1.getFeature(key)
        friend2Value = friend2.getFeature(key)
        if friend1Value == friend2Value:
            number_of_common_attributes += 1
            
    return (number_of_common_attributes + 1)


def simiarity_according_to_attributes_and_friendship( persons, friend1ID, friend2ID ):
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

