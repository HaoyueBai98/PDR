from torch.utils.data import Dataset

from utils.datacollection.Operation import *

class FTDataset(Dataset):
    def __init__(self, name, padding=True):
        super(FTDataset, self).__init__()

    def __getitem__(self, item):
        pass


