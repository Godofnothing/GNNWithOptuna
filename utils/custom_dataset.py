import torch
import numpy as np

from torch.utils.data import Dataset
from torch_geometric.data import Data


class CustomDataset(Dataset):

    def __init__(self, data_path: str) -> None:
        super().__init__()
        # load .npz data
        npz_data = np.load(data_path)

        self.data = Data(
            x=torch.from_numpy(npz_data['node_features']),
            y=torch.from_numpy(npz_data['node_labels']),
            edge_index=torch.from_numpy(npz_data['edges']).T,
            train_mask=torch.from_numpy(npz_data['train_masks']).T,
            val_mask=torch.from_numpy(npz_data['val_masks']).T,
            test_mask=torch.from_numpy(npz_data['test_masks']).T,
        )

    def __len__(self):
        return 1

    def __getitem__(self, idx) -> Data:
        assert idx == 0, "There is only one item in the dataset"
        return self.data
        