"""
Graph Mining - ALTEGRAD - Dec 2018
"""

# Import modules
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


############## Question 1
# Load the graph into an undirected NetworkX graph


##################
G = read_edgelist(path='../datasets/CA-HepTh.txt', comments='#', delimiter='\t', nodetype=int, create_using=nx.Graph())
##################


############## Question 2
# Network Characteristics


##################
print("The number of nodes of G is : ", G.number_of_nodes())
print("The number of edges of G is : ", G.number_of_edges())
print("The number of connected components of G is :", nx.number_connected_components(G))

# The graph is not connected so we will store the largest connected component subgraph

GCC = max(nx.connected_component_subgraphs(G), key=len)
print("The fraction of nodes in GCC is : ", GCC.number_of_nodes()/G.number_of_nodes())
print("The fraction of edges in GCC is : ", GCC.number_of_edges()/G.number_of_edges())

##################



############## Question 3
# Analysis of degree distribution


##################
degree_sequence = [d for n, d in G.degree()]
print("The minimum degree of the nodes of the graph is :", min(degree_sequence))
print("The maximum degree of the nodes of the graph is :", max(degree_sequence))
print("The mean degree of the nodes of the graph is :", np.mean(degree_sequence))
print("The median degree of the nodes of the graph is :", np.median(degree_sequence))

y = nx.degree_histogram(G)

plt.plot(y , 'b-', marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.show()

# Plot using log-log axis
plt.loglog(y , 'b-', marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.show()

##################
