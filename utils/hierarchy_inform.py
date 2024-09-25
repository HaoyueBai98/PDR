import torch

"""
this class is used to init model trainer[embedding]

sos : root 0
"""


class HierarchyStructure:
    def __init__(self, layers, applyindices, aim, iam, hidm, heidm):
        self.layer_num = layers
        self.num_per_layer = torch.zeros(self.layer_num,
                                         dtype=torch.long) + 1
        self.num_nodes = len(applyindices) + self.layer_num + 1 # all labels + 4 eos + 1 padding
        self.label_map = dict()  # store the applyid's indice in each layer
        self.applyid_index_map = aim
        self.index_applyid_map = iam
        self.hierarchical_label = torch.zeros(self.layer_num + 1, self.num_nodes,
                                              dtype=torch.long)  # 存放每一层的标签所代表的global label
        self.flat_init = False
        self.hierarchical_init = False
        self.sos_token = 0  # <\sos> is root
        self.pad_token = 1
        for i in applyindices:
            ids = self.index_applyid_map[i]
            if ids == 'root':
                continue
            else:
                self.num_per_layer[int((len(ids) - 1) / 2 + 1) - 1] += 1
        # self.num_per_layer[-1] = self.num_nodes
        for applyid, local_labels in hidm.items(): # hid is local-wise label
            heid = heidm[applyid] # hmid is global-wise label
            for layer, local_label in enumerate(local_labels[1:-1]):
                # -1 => layer_wise eos
                # -2 => padding
                self.hierarchical_label[layer][local_label] = heid[layer + 1]
        # last layer is all nodes => for global use
        self.hierarchical_label[-1] = torch.arange(0, self.num_nodes) # adding last layer
        self.hierarchical_label[:-1, -1] = self.pad_token

# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--id-label-map", type=str
    #                     , default='~/jupyter_base/HMTransformer/ds/data/label_map_2019',
    #                     help="label map, map the application id to the applyid code str")
    # parser.add_argument("-ii", "--indice-app-map", type=str
    #                     , default='~/jupyter_base/HMTransformer/ds/data/indice_id',
    #                     help="apply id map, map the applyid to the applyid index")
    # parser.add_argument("-ln", "--label-number", type=int, default=3545)
    #
    # # hierarchy data input
    # parser.add_argument("-hi", "--hierarchy-indice", type=str
    #                     , default='~/jupyter_base/HMTransformer/ds/data/hierarchical_indice_id')
    # parser.add_argument("-beta", "--beta", type=float, default=0.01,
    #                     help='to modified the weighted of global-wise train or layer-wise train')
    # args = parser.parse_args()
    # aim = dict()
    # iam = dict()
    # with open(args.indice_app_map, 'r') as f:
    #     mapping = [line.replace('\r', '').replace('\n', '').split('\t')
    #                for line in tqdm.tqdm(f, desc="Loading Indice apply id Map")]
    #     for line in mapping:
    #         indice = line[0]
    #         label = line[1]
    #         aim[label] = int(indice)
    #         iam[int(indice)] = label
    #
    # # vocab = ZHWordVocab('~/jupyter_base/science_embedding/data/corpus_2019')
    # vocab = ZHWordVocab.load_vocab('~/jupyter_base/HMTransformer/ds/data/vocab.save')
    # # vocab.save_vocab('~/jupyter_base/science_embedding/data/vocab.save')
    # ds = ZHSCIBERTDataset('~/jupyter_base/HMTransformer/ds/data/corpus_2019',
    #                       '~/jupyter_base/HMTransformer/ds/data/label_map_2019',
    #                       '~/jupyter_base/HMTransformer/ds/data/indice_id',
    #                       vocab=vocab, seq_len=50, neg_num=10,
    #                       hierarchical_path='~/jupyter_base/HMTransformer/ds/data/hierarchical_indice_id')
    #
    # aihs = HierarchyStructure(layers=4, applyindices=iam.keys(), aim=aim, iam=iam, heidm=ds.hemid_map,
    #                                  hidm=ds.hid_map)
    # # aihs.load_hierarchical_label(f=args.hierarchy_indice)
    # print(1)
