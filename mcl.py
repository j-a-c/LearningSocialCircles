import numpy


"""
Implementation of the Markov Cluster Algorithm

Person is the id of the person to calculate the cluster for.
Friend is the map of people to their friends.
Weights are the weights to use for the MCL matrx.
"""
def mcl(origPerson, data, weights=None):
    ITERATIONS = 20
    POWER = 2

    friendMap = data.friendMap

    # Create adjacency matrix. We will map ids to indices in the matrix.
    idMap = {}
    reverseIdMap = {}

    numFriends = len(friendMap[origPerson])
    idMap[origPerson] = numFriends

    markovMatrix = numpy.zeros((numFriends, numFriends))

    # Create person to index mappings
    currentId = 0
    for friend in friendMap[origPerson]:
        idMap[friend] = currentId
        reverseIdMap[currentId] = friend
        currentId += 1

    if weights:
        for friend in friendMap[origPerson]:
            # Add self loop
            markovMatrix[idMap[friend]][idMap[friend]] = 1
            # Add adjacent friends
            for neighbor in friendMap[friend]:
                if neighbor in idMap and neighbor != origPerson:
                    weight = weights[friend][neighbor]
                    markovMatrix[idMap[friend]][idMap[neighbor]] = weight
                    markovMatrix[idMap[neighbor]][idMap[friend]] = weight

    else:
        for friend in friendMap[origPerson]:
            # Add self loop
            markovMatrix[idMap[friend]][idMap[friend]] = 1
            # Add adjacent friends
            for neighbor in friendMap[friend]:
                if neighbor in idMap and neighbor != origPerson:
                    markovMatrix[idMap[friend]][idMap[neighbor]] = 1
                    markovMatrix[idMap[neighbor]][idMap[friend]] = 1
            # Add missing connections
            totalMissing = 0
            for column in range(numFriends):
                if markovMatrix[idMap[friend]][column] == 0:
                    totalMissing += 1
            for column in range(numFriends):
                if markovMatrix[idMap[friend]][column] == 0:
                    markovMatrix[idMap[friend]][column] = 1.0 / totalMissing

    # Start MCL algorithm
    for iteration in range(ITERATIONS):
        # expand matrix
        markovMatrix = numpy.linalg.matrix_power(markovMatrix,2)
        # inflate columns
        for col in range(numFriends):
            origCol = markovMatrix[:,col]
            for index in range(numFriends):
                origCol[index] = (origCol[index])**POWER
            total = sum(origCol)
            if total == 0: total = 1
            for index in range(numFriends):
                origCol[index] /= total
            # Update original matrix column
            for row in range(numFriends):
                markovMatrix[row, col] = origCol[row]

    # Find circles
    threshold = 0.0001
    circles = []
    for row in range(numFriends):
        if max(markovMatrix[row,:]) > threshold:
            newCircle = []
            for index in range(numFriends):
                if markovMatrix[row,index] > threshold:
                    newCircle.append(reverseIdMap[index])

            if len(newCircle) > 1:
                circles.append(newCircle)

    return circles
