###
# Post-prediction cluster pruning methods. All of these functions should accept
# data, a person, and a list of clusters belonging to the person. The return
# type is a list of clusters returning to the person.
###

def noPrune(data, person, clusters):
    return clusters
