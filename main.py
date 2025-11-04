import networkx as nx

# Load the directed graph from AmazonGraph.txt
G = nx.read_edgelist('AmazonGraph.txt', 
                     create_using=nx.DiGraph(), 
                     nodetype=int,
                     data=False)

print(f"Graph loaded successfully!")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
