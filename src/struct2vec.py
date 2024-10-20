# src/struct2vec.py

import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.nn import Node2Vec as PyGNode2Vec

class Struct2Vec:
    def __init__(self, repo_path, vector_size):
        self.repo_path = repo_path
        self.vector_size = vector_size

    def generate(self):
        directory_structure = self.get_directory_structure()
        G = nx.DiGraph(directory_structure)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PyGNode2Vec(G, embedding_dim=self.vector_size, walk_length=30, context_size=10, walks_per_node=200)
        loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for epoch in range(100):
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            node_embeddings = model(torch.arange(G.number_of_nodes(), device=device)).cpu().numpy()

        return np.mean(node_embeddings, axis=0)

    def get_directory_structure(self):
        structure = {}
        for root, dirs, files in os.walk(self.repo_path):
            structure[root] = {}
            for d in dirs:
                structure[root][d] = None
            for f in files:
                structure[root][f] = None
        return structure
