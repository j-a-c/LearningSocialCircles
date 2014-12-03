import collections
import networkx as nx
import os
import pylab


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

    def plotGraph(self, graph, nodeList, nodeColors, title, save, show):
        OUTPUT_DIR = 'graphs'
        
        # Use spring layout for nice format
        pos = nx.spring_layout(graph)
        pylab.figure()
        
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

    def createGraph(self, graph, title, origPerson, show, save):
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
                
        self.plotGraph(graph, nodeList, nodeColors, title, save, show)
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
        
    def visualizeClusters(self, clusters, title='Cluster', save=False, show=True):
        # Colors we support
        colors = ['blue', 'green', 'yellow', 'orange', 'purple',
                'indigo', 'magenta', 'pink', 'crimson', 'firebrick', 'chartreuse',
                'gold', 'mistyrose', 'lightsalmon2', 'peru', 'limegreen',
                'khaki1', 'grey73', 'darkorange2', 'cadetblue4', 'aliceblue',
                'mediumseagreen']
        cluster_number = 0
        graph = nx.Graph()
        
        for clusterID in clusters:
            for friend in clusters[clusterID]:
                graph.add_node(friend, style='filled', fillcolor=colors[cluster_number%len(colors)])
                if clusterID != friend:
                    graph.add_edge(clusterID, friend)
            cluster_number += 1
        
        # Convert data to pylab input
        nodeColors = []
        nodeList = []
        for nodeInfo in graph.nodes(data=True):
            nodeList.append(nodeInfo[0])
            nodeColors.append(nodeInfo[1]['fillcolor'])
                
        self.plotGraph(graph, nodeList, nodeColors, title, save, show)
            
    
