import torch
from dgl.nn import GATConv
import torch.nn as nn

class GATEncoder(nn.Module):
    def __init__(self, layer_num, in_feats, hidden_feat, out_feats, num_heads,
                 feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, activation=None):
        super(GATEncoder, self).__init__()
        assert layer_num > 1
        self.gat_layer = nn.ModuleList()
        if layer_num == 1:
            self.gat_layer.append(
                GATConv(in_feats, out_feats, num_heads, feat_drop=feat_drop, attn_drop=attn_drop,
                        negative_slope=negative_slope, residual=residual, activation=activation, allow_zero_in_degree=False))
        else:
            self.gat_layer.append(
                GATConv(in_feats, hidden_feat, num_heads, feat_drop=feat_drop, attn_drop=attn_drop,
                        negative_slope=negative_slope, residual=residual, activation=activation, allow_zero_in_degree=False))
            if layer_num > 2:
                for i in range(layer_num - 2):
                    self.gcn_layer.append(
                        GATConv(hidden_feat, hidden_feat, num_heads, feat_drop=feat_drop, attn_drop=attn_drop,
                        negative_slope=negative_slope, residual=residual, activation=activation, allow_zero_in_degree=False))
            self.gat_layer.append(
                GATConv(hidden_feat, out_feats, num_heads, feat_drop=feat_drop, attn_drop=attn_drop,
                        negative_slope=negative_slope, residual=residual, activation=activation, allow_zero_in_degree=False))

    def forward(self, blocks, feat, weight=None, mode='sample'):
        if mode == 'sample':
            for index, layer in enumerate(self.gat_layer):
                feat = layer(blocks[index], feat)
                feat = feat.mean(1)
        else:
            for layer in self.gat_layer:
                feat = layer(blocks, feat)
                feat = feat.mean(1)
        return feat