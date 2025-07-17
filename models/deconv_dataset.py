from torch.utils.data import Dataset
import numpy as np

from models.utils import conver_adata_X_to_numpy


class Spatial_Exp_Dataset(Dataset):
    def __init__(self, data, neighbor_index):
        self.data = conver_adata_X_to_numpy(data)
        self.len = self.data.shape[0]
        self.neighbor_index = np.array(neighbor_index, dtype=np.int32)

    def __getitem__(self, index):
        data = self.data[index, :]
        neighbor_index = self.neighbor_index[index, :]
        return index, data, neighbor_index

    def __len__(self):
        return self.len