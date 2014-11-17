import collections
import networkx as nx
import pylab

def visualize(data):

    print 'Starting visualization.'

    # Color for nodes in multiple groups (need to fix muli-coloring)
    MULTICOLOR = 'white'
    # Color for nodes in no groups
    NO_CIRCLE = 'black'
    # Edge color (somthing light)
    EDGE_COLOR = '#eeefff'
    # Color for the original person
    ORIG_COLOR = 'red'

    print 'The original person is colored:', ORIG_COLOR
    print 'Nodes in multiple circles are colored:', MULTICOLOR
    print 'Nodes in no circles are colored:', NO_CIRCLE
    print 'Edge colors are:', EDGE_COLOR

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
        for friend in personToColor:
        #for friend in trainingMap[origPerson][0]:
            friendColorList = personToColor[friend]
            # Person is only in one circle
            if len(friendColorList) == 1:
                graph.add_node(friend, style='filled', fillcolor=colors[friendColorList[0]])
            # Person is in multiple circle, color them differently.
            else:
                graph.add_node(friend, style='filled', fillcolor=MULTICOLOR)

            # Add edges
            for friendsFriend in friendMap[friend]:
                # This condition manually adds friends that are not in circles.
                if (friendsFriend != origPerson) and not (friendsFriend in personToColor) and not (friendsFriend in friendNotInCircles):
                    friendNotInCircles.append(friendsFriend)

                # This is the criteria for adding a edge.
                graph.add_edge(friend, friendsFriend)

        # Use spring layout for nice format
        pos = nx.spring_layout(graph)
        pylab.figure(1)
        # Convert data to pylab input
        nodeColors = []
        for nodeInfo in graph.nodes(data=True):
            if 'fillcolor' in nodeInfo[1]:
                color = nodeInfo[1]['fillcolor']
                nodeColors.append(color)
            elif nodeInfo[0] == origPerson:
                nodeColors.append(ORIG_COLOR)
            else:
                nodeColors.append(NO_CIRCLE)
        # Draw pylab compatible graph
        nx.draw_networkx(graph, pos, nodelist=graph.nodes(),
                node_color=nodeColors, with_labels=False, node_size=60,
                edge_color=EDGE_COLOR)
        # Add title
        pylab.title(origPerson)
        pylab.show()

    print 'Ending visualization.'
