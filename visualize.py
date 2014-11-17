import collections
import networkx as nx
import os
import pylab

def visualize(data):

    OUTPUT_DIR = 'graphs'

    print 'Starting visualization.'

    MULTICOLOR = 'white'
    NO_CIRCLE = 'black'

    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple',
            'indigo', 'magenta', 'pink', 'crimson', 'firebrick', 'chartreuse',
            'gold', 'mistyrose2', 'lightsalmon2', 'peru', 'limegreen',
            'khaki1', 'grey73', 'darkorange2', 'cadetblue4', 'aliceblue',
            'mediumseagreen']

    numColors = len(colors)
    print 'Number of colors:', numColors

    trainingMap = data.trainingMap
    friendMap = data.friendMap

    # Map people to certain colors
    for origPerson in trainingMap:
        friendNotInCircles = []
        personToColor = collections.defaultdict(list)

        # Assign colors to people
        colorIndex = 0
        for circle in trainingMap[origPerson]:
            for friend in circle:
                personToColor[friend].append(colorIndex)
            colorIndex += 1

            if colorIndex > numColors:
                break

        # Construct graph
        graph = nx.Graph()
        for friend in personToColor:
        #for friend in trainingMap[origPerson][0]:
            friendColorList = personToColor[friend]
            # Person is only in one circle
            if len(friendColorList) == 1:
                graph.add_node(friend, style='filled', fillcolor=colors[friendColorList[0]])
            # Person is in multiple circle, color them differently.
            else:
                graph.add_node(friend, style='filled', fillcolor=MULTICOLOR)

            for friendsFriend in friendMap[friend]:
                if (friendsFriend != origPerson) and not (friendsFriend in personToColor) and not (friendsFriend in friendNotInCircles):
                    friendNotInCircles.append(friendsFriend)

                graph.add_edge(friend, friendsFriend)

        # Draw graph
        #output = nx.to_agraph(graph)
        #output.layout()
        #outputPath = os.path.join(OUTPUT_DIR, origPerson + '.png')
        #output.draw(outputPath)
        pos = nx.spring_layout(graph)
        pylab.figure(1)
        nodeColors = []
        for nodeInfo in graph.nodes(data=True):
            if 'fillcolor' in nodeInfo[1]:
                color = nodeInfo[1]['fillcolor']
                nodeColors.append(color)
            else:
                nodeColors.append(NO_CIRCLE)
        nx.draw_networkx(graph, pos, nodelist=graph.nodes(),
                node_color=nodeColors, with_labels=False, node_size=60)
        #outputPath = os.path.join(OUTPUT_DIR, origPerson + '.svg')
        #pylab.savefig(outputPath)
        #pylab.figure(3, figsize=(12,12))
        #nx.draw(graph, pos)
        pylab.title(origPerson)
        pylab.show()

    print 'Ending visualization.'
