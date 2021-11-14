import math
import random
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import numpy as np

import dgl
import torch

from utils import adj_tensorlize
from dgl.data import CoraGraphDataset


class BaseGraph:
    def __init__(self, graph_name = "test"):
        self.graph_list = ["test", "cora"]
        self.graph = self.build_graph_cora(graph_name)
        self.adj = self.get_adj(self.graph)
        self.features = self.graph.ndata["feat"]
        self.num_nodes = self.graph.num_nodes()
        self.node_index = list(range(self.num_nodes))
        self.num_edges = self.graph.num_edges()
        self.nodes = self.graph.nodes()
        
        #self.features = self.init_node_feat(self.graph)

    def build_graph_cora(self, graph):
        # Default: ~/.dgl/ 
        if graph == "test":
            graph = self.build_graph_test()
        elif graph == "cora":
            data = CoraGraphDataset()
            graph = data[0]
        else:
            raise ValueError("Unknow graph type : {}, graph list: {}".format(graph, " ".join(self.graph_list)))

        return graph
    
    def build_graph_test(self):
        """a demo graph: just for graph test
        """
        src_nodes = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6])
        dst_nodes = torch.tensor([1, 2, 0, 2, 0, 1, 3, 4, 5, 6, 2, 3, 3, 3])
        graph = dgl.graph((src_nodes, dst_nodes))
        # edges weights if edges has else 1
        graph.edata["w"] = torch.ones(graph.num_edges())
        return graph
    
    def convert_symmetric(self, X, sparse=True):
        # add symmetric edges
        if sparse:
            X += X.T - sp.diags(X.diagonal())
        else: 
            X += X.T - np.diag(X.diagonal())
        return X
        
    def add_self_loop(self, graph):
        # add self loop
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        return graph

    def get_adj(self, graph):
        graph = self.add_self_loop(graph)
        # edges weights if edges has weights else 1
        graph.edata["w"] = torch.ones(graph.num_edges())
        adj = coo_matrix((graph.edata["w"], (graph.edges()[0], graph.edges()[1])),
                         shape=(graph.num_nodes(), graph.num_nodes()))

        #  add symmetric edges
        #adj = self.convert_symmetric(adj, sparse=True)
        # adj normalize and transform matrix to torch tensor type
        adj = adj_tensorlize(adj, is_sparse=True)

        return adj

    def init_node_feat(self, graph):
        # init graph node features
        self.nfeat_dim = graph.number_of_nodes()

        row = list(range(self.nfeat_dim))
        col = list(range(self.nfeat_dim))
        indices = torch.from_numpy(
            np.vstack((row, col)).astype(np.int64))
        values = torch.ones(self.nfeat_dim)

        features = torch.sparse.FloatTensor(indices, values,
                                            (self.nfeat_dim, self.nfeat_dim))
        return features

class MotifAugmentedNet(BaseGraph):
    def __init__(self, graph_name="test"):
        super().__init__(graph_name=graph_name)
        self.motif_list = self.get_motif_from_adj()
        self.motif_index = list(range(self.num_nodes, self.num_nodes + len(self.motif_list)))
        self.num_motifs = len(self.motif_list)
        self.motif_start_id = self.num_nodes
        self.num_nodes = self.num_nodes + self.num_motifs
        self.features = self.get_augmented_features() 
        self.num_edges ,self.adj = self.get_augmented_adj()
        #self.num_nodes
    
    def get_motif_from_adj(self):
        motif_list = []
        for node0 in range(len(self.adj)):
            for node1 in self.adj[node0]._indices()[0]:
                if node1 <= node0:
                    continue
                for node2 in self.adj[node1]._indices()[0]:
                    if node2 <=node0 or node2 <= node1:
                        continue
                    if node2 in self.adj[node0]._indices()[0]:
                        motif_list.append([node0, int(node1), int(node2)])
        return motif_list
    
    def get_augmented_features(self):
        motif_features = []
        for motif in self.motif_list:
            motif_features.append(torch.mean(torch.stack([self.features[motif[0]], self.features[motif[1]], self.features[motif[2]]]), dim = 0))
        motif_features = torch.stack(motif_features)
        augmented_features = torch.cat([self.features, motif_features])
        return augmented_features

    def get_augmented_adj(self):
        motif_src_nodes = []
        motif_dst_nodes = []
        for node_index, motif in zip(self.motif_index, self.motif_list):
            neighbor_set = set()
            for node in motif:
                for neighbor in self.adj[node]._indices()[0]:
                    neighbor_set.add(int(neighbor))
                neighbor_set.add(node)
            motif_dst_nodes.append(torch.tensor(list(neighbor_set)))
            motif_src_nodes.append(torch.tensor([node_index]*len(neighbor_set)))
        motif_src_nodes = torch.cat(motif_src_nodes)
        motif_dst_nodes = torch.cat(motif_dst_nodes)

        src_nodes = torch.cat([self.graph.edges()[0], motif_src_nodes, motif_dst_nodes])
        dst_nodes = torch.cat([self.graph.edges()[1], motif_dst_nodes, motif_src_nodes])

        augmented_num_edges = self.num_edges + 2*len(motif_src_nodes)
        w = torch.ones(augmented_num_edges)
        adj = coo_matrix((w, (src_nodes, dst_nodes)),
                         shape=(self.num_nodes, self.num_nodes))
        adj = adj_tensorlize(adj, is_sparse=True)
        return augmented_num_edges, adj

class AnomalyMotifAugmentedNet(MotifAugmentedNet):
    def __init__(self, graph_name="test", p1 = 0.01, p2 = 0.03):
        super().__init__(graph_name=graph_name)
        self.p1 = p1
        self.p2 = p2

        self.adj, self.structure_anomaly_nodes, self.structure_anomaly_motifs = self.add_structure_anomaly()

        self.attribute_anomaly_nodes, self.attribute_anomaly_motifs = self.add_attribute_anomaly()
    
    def add_structure_anomaly(self):
        structure_anomaly_nodes_num = int(self.num_motifs * self.p1)
        structure_anomaly_motif_num = int(self.num_motifs * self.p2)

        structure_anomaly_nodes = random.sample(self.node_index, structure_anomaly_nodes_num)
        structure_anomaly_motifs = random.sample(self.motif_index, structure_anomaly_motif_num)
        anomaly_adj = self.adj.to_dense()
        
        for index in structure_anomaly_nodes:
            anomaly_adj[index][:] = 0
            anomaly_adj[:][index] = 0
            anomaly_adj[index][index] = 1
        
        for index in structure_anomaly_motifs:
            anomaly_adj[index][:] = 0
            anomaly_adj[:][index] = 0
            anomaly_adj[index][index] = 1
            # for sub_node in self.motif_list[index - self.motif_start_id]:
            #     anomaly_adj[index][int(sub_node)] = 1
            #     anomaly_adj[int(sub_node)][index] = 1
        anomaly_adj = anomaly_adj.to_sparse()
        return anomaly_adj, structure_anomaly_nodes, structure_anomaly_motifs

    def add_attribute_anomaly(self):
        #attribute_anomaly_nodes = random.sample(self.node_index, int(self.num_nodes * self.p1))
        attribute_anomaly_nodes_num = int(self.num_motifs * self.p1)
        attribute_anomaly_motif_num = int(self.num_motifs * self.p2)
        attribute_anomaly_nodes = []

        attribute_anomaly_motifs = random.sample(self.motif_index, attribute_anomaly_motif_num)

        for index in attribute_anomaly_motifs:
            candi_indexs = random.sample(self.motif_index, 50)
            far_motif_feature, far_motif_index = self.get_far_feature(candi_indexs, index)
            self.features[index] = far_motif_feature
            motif_nodes = self.motif_list[index - self.graph.num_nodes()]
            far_motif_nodes = self.motif_list[far_motif_index - self.graph.num_nodes()]
            for i in range(3):
                self.features[motif_nodes[i]] = self.features[far_motif_nodes[i]]
                attribute_anomaly_nodes.append(motif_nodes[i])
        if attribute_anomaly_motif_num*3 < attribute_anomaly_nodes_num:
            sample_more_nodes = random.sample(self.node_index, attribute_anomaly_nodes_num-attribute_anomaly_motif_num*3)
            attribute_anomaly_nodes.extend(sample_more_nodes)
            for index in sample_more_nodes:
                candi_indexs = random.sample(self.node_index, 50)
                far_node_feature, far_node_index = self.get_far_feature(candi_indexs, index)
                self.features[index] = far_node_feature
        return attribute_anomaly_nodes, attribute_anomaly_motifs


    def get_far_feature(self, candi_indexs, index):
        origin_feat = self.features[index]
        max_dist = 0
        for candi_index in candi_indexs:
            dist = torch.dist(origin_feat, self.features[candi_index], p=2)
            if dist > max_dist:
                max_dist = dist
                anomaly_index = index
        return self.features[anomaly_index], anomaly_index
    




            






    

    



        

