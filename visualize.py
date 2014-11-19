import collections
import networkx as nx
import os
import pylab

"""
Returns true if an edge exists between the two people in the orginal topology.
"""
def originalTopology(data, person1, person2):
    if person1 in data.friendMap[person2]:
        return True
    return False


def friendsInCommon(data, person1, person2, threshold=1):
    numFriendsInCommon = 0
    for friend in data.friendMap[person1]:
        if friend in data.friendMap[person2]:
            numFriendsInCommon += 1
    return numFriendsInCommon > threshold


"""
Returns true if the two people have more than 'threshold' attributes in common.
Does not include the amount of friends in common.
"""
def similarAttributes(data, person1, person2, threshold=3):
    numberAttributesInCommon = 0
    for key in data.featureMap[person1]:
        person1Value = data.featureMap[person1][key]
        person2Value = None
        if key in data.featureMap[person2]:
            person2Value = data.featureMap[person2][key]
        if person1Value == person2Value:
            numberAttributesInCommon += 1
    if numberAttributesInCommon > threshold:
        return True
    return False

def topologyAndAttributes(data, person1, person2):
    return similarAttributes(data, person1, person2) or friendsInCommon(data,
            person1, person2)

class Visualizer():
    def __init__(self):
        # Color for nodes in multiple groups (need to fix muli-coloring)
        self.MULTICOLOR = 'white'
        # Color for nodes in no groups
        self.NO_CIRCLE = 'black'
        # Edge color (somthing light)
        self.EDGE_COLOR = '#eeefff'
        # Color for the original person
        self.ORIG_COLOR = 'red'

    def createGraph(self, graph, title, origPerson, show, save):
        OUTPUT_DIR = 'graphs'
        # Use spring layout for nice format
        pos = nx.spring_layout(graph)
        pylab.figure(1)
        # Convert data to pylab input
        nodeColors = []
        nodeList = []
        for nodeInfo in graph.nodes(data=True):
            nodeList.append(nodeInfo[0])
            if 'fillcolor' in nodeInfo[1]:
                color = nodeInfo[1]['fillcolor']
                nodeColors.append(color)
            elif nodeInfo[0] == origPerson:
                nodeColors.append(self.ORIG_COLOR)
            else:
                nodeColors.append(self.NO_CIRCLE)
        # Draw pylab compatible graph
        nx.draw_networkx(graph, pos, nodelist=nodeList,
                node_color=nodeColors, with_labels=False, node_size=60,
                edge_color=self.EDGE_COLOR)
        # Add title
        pylab.title(title)
        if save:
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            savePath = os.path.join(OUTPUT_DIR, title + '.png')
            pylab.savefig(savePath)
        if show:
            pylab.show()


    """
    Visualizes a friend map given a data pack and function which determines if an
    edge exists between two people. The default edge function is to use the
    original graph topology.

    data is data pack from Kaggle.
    edgefunc is a return that takes the data and two people and returns true if an
        edge should exist between the two people.
    show specifies whether to show the resulting graph.
    save specifies whether to save the resulting graph.
    """
    def visualize(self, data, edgefunc, split=False, show=True, save=False):

        print 'Starting visualization.'

        print 'The original person is colored:', self.ORIG_COLOR
        print 'Nodes in multiple circles are colored:', self.MULTICOLOR
        print 'Nodes in no circles are colored:', self.NO_CIRCLE
        print 'Edge colors are:', self.EDGE_COLOR

        # Colors we support
        colors = ['blue', 'green', 'yellow', 'orange', 'purple',
                'indigo', 'magenta', 'pink', 'crimson', 'firebrick', 'chartreuse',
                'gold', 'mistyrose', 'lightsalmon2', 'peru', 'limegreen',
                'khaki1', 'grey73', 'darkorange2', 'cadetblue4', 'aliceblue',
                'mediumseagreen']

        numColors = len(colors)
        print 'Number of colors:', numColors

        trainingMap = data.trainingMap
        friendMap = data.friendMap

        # Map people to certain colors
        for origPerson in trainingMap:
            if split:
                # Assign colors to people
                colorIndex = 0
                circleIndex = 0
                for circle in trainingMap[origPerson]:
                    circleIndex += 1
                    friendNotInCircles = []
                    personToColor = collections.defaultdict(list)

                    for friend in circle:
                        personToColor[friend].append(colorIndex)

                    # Construct graph
                    graph = nx.Graph()
                    # Add colors to people and add missing colors
                    for friend in personToColor:
                        friendColorList = personToColor[friend]
                        # Person is only in one circle
                        graph.add_node(friend, style='filled', fillcolor=colors[friendColorList[0]])

                        # Add edges
                        for friendsFriend in friendMap[friend]:
                            # This condition manually adds friends that are not in circles.
                            if (friendsFriend != origPerson) and not (friendsFriend in personToColor) and not (friendsFriend in friendNotInCircles):
                                friendNotInCircles.append(friendsFriend)

                    # Attempt to add edges between all pairs of people
                    for friend in personToColor:
                        for friendsFriend in friendMap[friend]:
                            if edgefunc(data, friend, friendsFriend):
                                graph.add_edge(friend, friendsFriend)
                    for person1 in friendMap[origPerson]:
                        for person2 in friendMap[origPerson]:
                            if edgefunc(data, person1, person2):
                                graph.add_edge(person1, person2)
                    for person1 in friendMap[origPerson]:
                        if edgefunc(data, origPerson, person1):
                            graph.add_edge(origPerson, person1)

                    graphTitle = origPerson + ', Circle ' + str(circleIndex) + '_' + str(len(trainingMap[origPerson]))
                    self.createGraph(graph, graphTitle, origPerson, show, save)
            else:
                friendNotInCircles = []
                personToColor = collections.defaultdict(list)

                # Assign colors to people
                colorIndex = 0
                for circle in trainingMap[origPerson]:
                    for friend in circle:
                        personToColor[friend].append(colorIndex)
                    colorIndex += 1
                    # If too many circles, break out
                    if colorIndex > numColors:
                        break

                # Construct graph
                graph = nx.Graph()
                # Add colors to people and add missing colors
                for friend in personToColor:
                    friendColorList = personToColor[friend]
                    # Person is only in one circle
                    if len(friendColorList) == 1:
                        graph.add_node(friend, style='filled', fillcolor=colors[friendColorList[0]])
                    # Person is in multiple circle, color them differently.
                    else:
                        graph.add_node(friend, style='filled', fillcolor=self.MULTICOLOR)

                    # Add edges
                    for friendsFriend in friendMap[friend]:
                        # This condition manually adds friends that are not in circles.
                        if (friendsFriend != origPerson) and not (friendsFriend in personToColor) and not (friendsFriend in friendNotInCircles):
                            friendNotInCircles.append(friendsFriend)


                # Attempt to add edges between all pairs of people
                for friend in personToColor:
                    for friendsFriend in friendMap[friend]:
                        if edgefunc(data, friend, friendsFriend):
                            graph.add_edge(friend, friendsFriend)
                for person1 in friendMap[origPerson]:
                    for person2 in friendMap[origPerson]:
                        if edgefunc(data, person1, person2):
                            graph.add_edge(person1, person2)
                for person1 in friendMap[origPerson]:
                    if edgefunc(data, origPerson, person1):
                        graph.add_edge(origPerson, person1)

                graphTitle = origPerson + ', All circles'
                self.createGraph(graph, graphTitle, origPerson, show, save)

        print 'Ending visualization.'
