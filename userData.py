
class Person(object):
    def __init__(self, personID ):
        self._personID = personID
        self._friends = set()
        self._features = {}
        
    def getFriends(self):
        return self._friends
    
    def getFeatures(self):
        return self._features
    
    def addFriend(self, friendID):
        return self._friends.add(friendID)
    
    def addFeature(self, key, value):
        self._features[key] = value
        
    def getFeature(self, key):
        return self._features.get(key, None)
    
    def isFriend(self, friendID):
        return (friendID in self._friends)
    
    
class Persons(object):
    def __init__(self):
        self._persons = {}
        self._originalPersons = []
        
    def getPerson(self, person_ID):
        if person_ID not in self._persons:
            self._persons[person_ID] = Person(person_ID)
        
        return self._persons[person_ID]
    
    def getAllPersons(self):
        return self._persons
        
    def addOriginalPerson(self, originalPersonID):
        self._originalPersons.append(originalPersonID)
        
    def getOriginalPersons(self):
        return self._originalPersons
    