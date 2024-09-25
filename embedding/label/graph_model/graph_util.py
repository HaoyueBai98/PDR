import dgl
import numpy
import torch
import tqdm

import inspect
import matplotlib.pyplot as plt
# import seaborn as sns


def get_layer(s):
    if s == 'root':
        return 0
    return int((len(s) - 1) / 2) + 1  # ret = 0 to 4, ret_reverse = 5 to 9


def build_graph(file, func, **kwargs):
    label_map = kwargs['label_map']
    with open(file, mode='r', encoding=kwargs['encoding']) as f:
        lines = [str(line).replace('\r', '').replace('\n', '').split('\t') for line in
                 tqdm.tqdm(f, desc="Loading Edges From File")]
        lines = [[int(label_map[line[0]]) + get_layer(line[0]),
                  int(label_map[line[1]]) + get_layer(line[1]), get_layer(line[0])] for line in
                 tqdm.tqdm(lines, desc="Processing Lines Init")]
    lines = numpy.asarray(lines)
    edge_type = lines[:, 2]
    lines = lines[:, :2]
    re_edge_type = edge_type + 5
    args = inspect.getfullargspec(func).args
    func_args = dict()
    for i in args:
        if kwargs.__contains__(i):
            func_args[i] = kwargs[i]
    _g = func(edges=lines, **func_args)
    _g.edata['etype'] = torch.from_numpy(numpy.concatenate([edge_type, re_edge_type]))
    return _g


def build_adj_matrix(file, label_map, weigh_th=0.0, num_node=2565, device=None):
    with open(file, mode='r') as f:
        lines = [str(line).replace('\r', '').replace('\n', '').split('\t') for line in
                 tqdm.tqdm(f, desc="Loading Edges From File")]
        edges = [[int(label_map[line[0]]) + int((len(line[0]) - 1) / 2) + 1,
                  int(label_map[line[1]]) + int((len(line[1]) - 1) / 2) + 1] for line in
                 tqdm.tqdm(lines, desc="Processing Lines Init")]
        weight = numpy.asarray([float(line[2]) for line in tqdm.tqdm(lines, desc="Processing Lines Init")])
    adj_matrix = torch.eye(num_node, device=device)
    num = 0
    for index, edge in enumerate(edges):
        if 0 < weigh_th:
            if weight[index] > weigh_th:
                num += 1
                adj_matrix[edge[0], edge[1]] = weight[index]
        else :
            num += 1
            adj_matrix[edge[0], edge[1]] = weight[index]
    print(num)
    return adj_matrix


def build_weight_graph(file, func, weigh_th=0.0, bi=False, **kwargs):
    label_map = kwargs['label_map']
    with open(file, mode='r', encoding=kwargs['encoding']) as f:
        lines = [str(line).replace('\r', '').replace('\n', '').split('\t') for line in
                 tqdm.tqdm(f, desc="Loading Edges From File")]
        # edges = [[int(label_map[line[0]]) + int((len(line[0]) - 1) / 2) + 1,
        #           int(label_map[line[1]]) + int((len(line[1]) - 1) / 2) + 1] for line in
        #          tqdm.tqdm(lines, desc="Processing Lines Init")]
        edges = [[int(label_map[line[1]]) + int((len(line[1]) - 1) / 2) + 1,
                  int(label_map[line[0]]) + int((len(line[0]) - 1) / 2) + 1] for line in
                 tqdm.tqdm(lines, desc="Processing Lines Init")]
        weight = numpy.asarray([float(line[2]) for line in tqdm.tqdm(lines, desc="Processing Lines Init")])
    if weigh_th > 0:
        th_select = weight > weigh_th
        weight = weight[th_select]
        edges = numpy.asarray(edges)[th_select]

    args = inspect.getfullargspec(func).args
    func_args = dict()
    for i in args:
        if kwargs.__contains__(i):
            func_args[i] = kwargs[i]
    _g = func(edges=edges, bi=bi, **func_args)
    if bi:
        _g.edata['weight'] = torch.from_numpy(numpy.append(numpy.asarray(weight), numpy.asarray(weight))).float()
    else:
        _g.edata['weight'] = torch.from_numpy(numpy.asarray(weight)).float()
    return _g


def build_homo_graph(edges, node_num, bi=True):
    # edges = numpy.asarray(edges)
    s_a = edges[:, 0]
    d_a = edges[:, 1]
    if bi:
        src = numpy.append(s_a, d_a)
        dst = numpy.append(d_a, s_a)
    else:
        src = s_a
        dst = d_a
    g = dgl.graph((src, dst), num_nodes=node_num)
    # g.add_self_loop()
    return g


def load_label_map(file):
    app_map = dict()
    indice_app_map = dict()
    with open(file, 'r') as f:
        mapping = [line.replace('\r', '').replace('\n', '').split('\t')
                   for line in tqdm.tqdm(f, desc="Loading Indice apply id Map")]
        for line in mapping:
            indice = int(line[1])  # apply indice
            label = line[0]  # applyid
            app_map[label] = indice
            indice_app_map[int(indice)] = label
    return app_map, indice_app_map


def weight_stat(file):
    with open(file, mode='r') as f:
        lines = [str(line).replace('\r', '').replace('\n', '').split('\t') for line in
                 tqdm.tqdm(f, desc="Loading Edges From File")]
        weight = numpy.asarray([float(line[2]) for
                                line in tqdm.tqdm(lines, desc="Processing Lines Init")])
    mu = numpy.mean(weight)
    sigma = numpy.std(weight)
    num = 1000
    normal_data = numpy.random.normal(mu, sigma, num)
    count, bins, ignored = plt.hist(weight, 100)
    logbins = numpy.logspace(numpy.log10(bins[0]), numpy.log10(bins[-1]), len(bins))
    count, bins, ignored = plt.hist(weight, bins=logbins)

    plt.plot(bins, 1 / (sigma * numpy.sqrt(2 * numpy.pi)) *
             numpy.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2,
             color='orange')
    plt.show()


if __name__ == '__main__':
    # weight_stat('~/jupyter_base/HMLC2022/data/raw_data/applyid_co_word_edge_count')
    # weight_stat('~/jupyter_base/HMLC2022/data/raw_data/applyid_co_word_edge_norm')
    app_map_path = '~/jupyter_base/HMLC2022/data/raw_data/applyid_indice_map'
    app_map, _ = load_label_map(app_map_path)
    print(build_adj_matrix(file='~/jupyter_base/HMLC2022/data/raw_data/applyid_co_word_edge_norm', label_map=app_map))
    # g = build_graph('~/jupyter_base/HMLC2022/data/raw_data/applyid_edge', build_homo_graph, node_num=2565,
    #                 encoding='utf-8', label_map=app_map)
    # weight_g = build_weight_graph('~/jupyter_base/HMLC2022/data/raw_data/applyid_co_word_edge_norm',
    #                               build_homo_graph, node_num=2565,
    #                               encoding='utf-8', label_map=app_map, weigh_th=0.1)
    # print(g)
    # print(g.edata['etype'])
    # # print(g.edges())
    # # print(app_map['C09'] + int((len('C09') - 1) / 2) + 1)
    # # print(app_map['C0902'] + int((len('C0902') - 1) / 2) + 1)
    # print(weight_g)
    # print(weight_g.edata['weight'])
    # count_g = build_weight_graph('~/jupyter_base/HMLC2022/data/raw_data/applyid_co_word_edge_count',
    #                              build_homo_graph, node_num=2565,
    #                              encoding='utf-8', label_map=app_map, weigh_th=0.1)
    # print(count_g)
    # print(count_g.edata['weight'])
