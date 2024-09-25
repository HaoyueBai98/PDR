import torch
import torch.nn as nn
from dgl.nn import GraphConv

from model.HMLC.utils.embedding.label.graph_model.graph_util import load_label_map, build_graph, build_weight_graph, \
    build_homo_graph


class GCNEncoder(nn.Module):
    def __init__(self, layer_num, in_feat, hidden_feat, out_feat, norm='both', weight=True, bias=True, activation=None):
        super(GCNEncoder, self).__init__()
        assert layer_num > 1
        self.gcn_layer = nn.ModuleList()
        if layer_num == 1:
            self.gcn_layer.append(
                GraphConv(in_feat, out_feat, norm=norm, weight=weight, bias=bias, activation=activation, allow_zero_in_degree=True))
        else:
            self.gcn_layer.append(
                GraphConv(in_feat, hidden_feat, norm=norm, weight=weight, bias=bias, activation=None, allow_zero_in_degree=True))
            if layer_num > 2:
                for i in range(layer_num - 2):
                    self.gcn_layer.append(
                        GraphConv(hidden_feat, hidden_feat, norm=norm, weight=weight, bias=bias, activation=None, allow_zero_in_degree=True))
            self.gcn_layer.append(
                GraphConv(hidden_feat, out_feat, norm=norm, weight=weight, bias=bias, activation=activation, allow_zero_in_degree=True))

    def forward(self, blocks, feat, weight=None, mode='sample', edge_weight=None):
        if mode == 'sample':
            for index, layer in enumerate(self.gcn_layer):
                feat = layer(blocks[index], feat, weight, edge_weight=edge_weight)
        else:
            for layer in self.gcn_layer:
                feat = layer(blocks, feat, weight, edge_weight=edge_weight)
        return feat

if __name__ == '__main__':
    app_map_path = '~/jupyter_base/HMLC2022/data/raw_data/applyid_indice_map'
    app_map, _ = load_label_map(app_map_path)
    g = build_graph('~/jupyter_base/HMLC2022/data/raw_data/applyid_edge', build_homo_graph, node_num=2566,
                    encoding='utf-8', label_map=app_map)
    print(list(app_map.values())[-1])
    model = GCNEncoder(layer_num=2, in_feat=64, hidden_feat=128, out_feat=64)
    feat = torch.randn((g.num_nodes(), 64))
    print(g)
    x = model.forward(g, feat, mode='norm')
    print(x)
    print(x.shape)
    w_model = GCNEncoder(layer_num=2, in_feat=64, hidden_feat=128, out_feat=64)
    weight_g = build_weight_graph('~/jupyter_base/HMLC2022/data/raw_data/applyid_co_word_edge_norm',
                                  build_homo_graph, node_num=2566,
                                  encoding='utf-8', label_map=app_map, weigh_th=0.1)
    y = w_model.forward(weight_g, feat, edge_weight=weight_g.edata['weight'], mode='norm')
    print(y)
    print(y.shape)

