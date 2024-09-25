import torch
import torch.nn as nn
import pickle

from model.embedding.label.graph_model.gnn.gnn_model import GCNEncoder


class LabelEmbedding(nn.Module):
    def __init__(self):
        super(LabelEmbedding, self).__init__()

    def forward(self, labels):
        pass


"""
随机初始化的节点表征
"""


class RandomLabelEmbedding(LabelEmbedding):
    def __init__(self, label_num, h_dim):
        super(RandomLabelEmbedding, self).__init__()
        embedding_tensor = torch.zeros((label_num, h_dim))
        nn.init.kaiming_uniform_(embedding_tensor)
        self.embedding = nn.Parameter(embedding_tensor)

    def forward(self, labels):
        return self.embedding[labels]


"""
一个可以从预训练中获取节点表征的embedding
"""


class PretrainedLabelEmbedding(LabelEmbedding):
    def __init__(self, init_file, id_file):
        super(PretrainedLabelEmbedding, self).__init__()
        embed_f = pickle.load(init_file)
        pass


"""
基于gnn的节点表征encoder
"""


class GraphLabelEmbedding(LabelEmbedding):
    def __init__(self, graph, in_dim, h_dim=0, out_dim=0, layer_num=0, graph_encoder=None, norm='both', weight=True,
                 bias=True,
                 activation=None, with_cuda=False, cuda_devices=None):
        super(GraphLabelEmbedding, self).__init__()
        if graph_encoder is None:
            self.graph_encoder = GCNEncoder(layer_num, in_dim, h_dim, out_dim, norm=norm, weight=weight, bias=bias,
                                            activation=activation)
        else:
            self.graph_encoder = graph_encoder
        embedding_tensor = torch.zeros((graph.num_nodes(), in_dim))
        nn.init.kaiming_uniform_(embedding_tensor)
        self.feat = nn.Parameter(embedding_tensor)
        if with_cuda:
            self.g = graph.to("cuda:{}".format(cuda_devices[0]))
        else:
            self.g = graph

    def forward(self, labels):
        feat = self.graph_encoder.forward(self.g, self.feat, weight=None, mode='all')
        return feat[labels]


if __name__ == '__main__':
    a = RandomLabelEmbedding(1000, 10)
    b = a.forward(torch.LongTensor([1]))
    print(b)
