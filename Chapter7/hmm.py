import numpy as np
import networkx as nx
import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt

states = ['A', 'B', 'C'] # A: Alpha-helix, B: Beta-sheet, C: Random
n_states = len(states)

transition_prob = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])

emission_prob = np.array([
    [0.6, 0.2, 0.2], 
    [0.2, 0.5, 0.3],
    [0.3, 0.3, 0.4] 
])

initial_prob = np.array([0.5, 0.3, 0.2]) # can be random

def generate_sequence(length):
    sequence = []
    state = np.random.choice(n_states, p=initial_prob)

    for _ in range(length):
        symbol = np.random.choice(n_states, p=emission_prob[state])
        sequence.append(states[state])
        state = np.random.choice(n_states, p=transition_prob[state])
    
    return sequence

sequence = generate_sequence(100)
print(sequence)

def generate_3d_coordinates(sequence):
    coordinates = np.cumsum(np.random.randn(len(sequence), 3), axis=0)
    return coordinates

coordinates = generate_3d_coordinates(sequence)
print(coordinates)

def create_graph(sequence, coordinates):
    G = nx.Graph()
    for i, (nucleotide, coord) in enumerate(zip(sequence, coordinates)):
        G.add_node(i, nucleotide=nucleotide, pos=coord)
    
    for i in range(len(sequence)-1):
        G.add_edge(i,i+1)
    
    return G

G = create_graph(sequence, coordinates)
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 5)])

pos = nx.spring_layout(G)  
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=16, font_color='black')
plt.show()

class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(3, 16)  # 3 features per node (x, y, z)
        self.conv2 = GCNConv(16, 3)  # Output 3 coordinates
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GNNModel()

def graph_to_data(G):
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = np.array([G.nodes[i]['pos'] for i in G.nodes], dtype=np.float32)
    x = torch.tensor(x, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index)

data = graph_to_data(G)
data = torch_geometric.data.Data(x=data.x, edge_index=data.edge_index)

def train(model, data, num_epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.x)  
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

train(model, data)