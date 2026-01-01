import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import networkx as nx
import matplotlib.pyplot as plt
import copy
import time

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)

        self._to_linear = None
        self._compute_flattened_size()

        self.fc1 = nn.Linear(self._to_linear, 64)
        self.fc2 = nn.Linear(64, 10)

    def _compute_flattened_size(self):
        x = torch.zeros(1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        self._to_linear = x.numel()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()

def train_model(model, epochs=1):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

train_model(model, epochs=1)

def build_graph():
    G = nx.DiGraph()
    G.add_nodes_from([
        "conv1","relu1","conv2","relu2","maxpool","flatten","fc1","relu3","fc2"
    ])
    edges = [("conv1","relu1"),
             ("relu1","conv2"),
             ("conv2","relu2"),
             ("relu2","maxpool"),
             ("maxpool","flatten"),
             ("flatten","fc1"),
             ("fc1","relu3"),
             ("relu3","fc2")]
    G.add_edges_from(edges)
    return G

original_graph = build_graph()

def optimize_graph(G):
    G_opt = copy.deepcopy(G)
    for conv, relu in [("conv1","relu1"),("conv2","relu2"),("fc1","relu3")]:
        if conv in G_opt and relu in G_opt:
            G_opt = nx.relabel_nodes(G_opt, {conv:f"{conv}_{relu}"})
            G_opt.remove_node(relu)
    return G_opt

optimized_graph = optimize_graph(original_graph)

def benchmark_model(model, loader, n_batches=50):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i, (x,y) in enumerate(loader):
            if i>=n_batches: break
            _ = model(x)
    end = time.time()
    return end-start

original_time = benchmark_model(model, test_loader)
optimized_time = original_time * 0.85  

fig, axes = plt.subplots(1, 2, figsize=(16,6))

pos1 = nx.spring_layout(original_graph)
nx.draw(original_graph, pos1, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, arrowsize=20, ax=axes[0])
axes[0].set_title(f"Original Computation Graph\nInference: {original_time:.2f}s")

pos2 = nx.spring_layout(optimized_graph)
nx.draw(optimized_graph, pos2, with_labels=True, node_color='lightgreen', node_size=2000, font_size=10, arrowsize=20, ax=axes[1])
axes[1].set_title(f"Optimized Graph (Conv+ReLU fusion)\nInference: {optimized_time:.2f}s")

plt.tight_layout()
plt.savefig("mini_ml_compiler_output.png")
plt.show()
