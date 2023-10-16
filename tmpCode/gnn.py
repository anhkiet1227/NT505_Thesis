#still bug
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv

# Load the dataset
dataset_file = 'synthetic_smart_contract_dataset.csv'
data = pd.read_csv(dataset_file)

# Assume you have a function to convert smart contract data to a graph representation
# Here's a dummy function to create a simple graph from the contract data
def create_graph(data):
    G = nx.DiGraph()
    # Add nodes and edges based on the contract structure
    # Modify this according to your specific dataset and contract structure
    # Example: G.add_node('function1'), G.add_edge('function1', 'function2')
    return G

# Convert each smart contract to a graph representation
graphs = [create_graph(contract_data) for contract_data in data['Smart Contract Code']]

# Compute the number of nodes in each graph
num_nodes = [len(graph.nodes) for graph in graphs]

# Define a simple GNN model for demonstration
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Prepare data for training (dummy node features and edge indices)
# Adjust this based on your actual node features and graph structure
# Here we assume dummy node features and edges
node_features = torch.randn(sum(num_nodes), 10)  # 10-dimensional node features

# Initialize and train the GNN model
gnn_model = GNNModel(input_dim=10, hidden_dim=32, output_dim=2)  # Adjust dimensions based on your problem
optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Dummy labels (0 or 1, indicating vulnerability presence or absence)
labels = torch.randint(0, 2, (len(graphs),))

# Train the GNN
for epoch in range(10):  # Adjust number of epochs
    for i in range(len(graphs)):
        # Generate random edge indices only if num_nodes > 0
        if num_nodes[i] > 0:
            edge_index = torch.randint(0, num_nodes[i], (2, 20))
        else:
            edge_index = torch.zeros((2, 20), dtype=torch.long)
        
        optimizer.zero_grad()
        output = gnn_model(node_features[:sum(num_nodes[:i+1])], edge_index)
        loss = criterion(output.unsqueeze(0), labels[i].squeeze()) # Reshape for correct batch size
        loss.backward()
        optimizer.step()

# Evaluate the GNN (dummy evaluation)
gnn_model.eval()
with torch.no_grad():
    # Perform inference on a new set of graphs
    # Adjust this based on your evaluation setup
    # Here we use the same dummy node features and edges for simplicity
    predicted_labels = []
    for i in range(len(graphs)):
        if num_nodes[i] > 0:
            edge_index = torch.randint(0, num_nodes[i], (2, 20))
            output = gnn_model(node_features[:sum(num_nodes[:i+1])], edge_index)
            predicted_labels.append(torch.argmax(output).item())
        else:
            # If num_nodes is 0, output a random label
            predicted_labels.append(torch.randint(0, 2, (1,)).item())

# Print the predicted labels
print("Predicted Labels:", predicted_labels)
print("Length of node_features:", len(node_features))
print("Dimensions of node_features:", node_features.size())
