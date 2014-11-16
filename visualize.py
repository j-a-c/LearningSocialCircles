import networkx as nx

def visualize(data):
    print 'Starting visualization.'

    graph = nx.Graph()

    for i in range(0, 100):
        graph.add_node(i, style='filled', fillcolor='red')
        graph.add_node(i+1, style='filled', fillcolor='yellow')
    for i in range(0, 100):
        graph.add_edge(i, (i+1) % 65)


    output = nx.to_agraph(graph)
    output.layout()
    output.draw('graph.png')

    print 'Ending visualization.'
