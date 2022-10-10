import torch
import pandas as pd

from utils import get_file_names, read_hdf5, find_target
from data_preparation import MelSpectrogram_v1

class G2NETDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        root: str = "data/raw/train/",
        labels_path: str = "data/raw/train_label/_labels.csv",
        min_max_sg_path: str = "data/infos/min_max_melspectrogram.csv"
    ):
        
        self.file_names = get_file_names(root)
        self.labels = pd.read_csv(labels_path)
        
        self.preparator = MelSpectrogram_v1(pd.read_csv(min_max_sg_path))
    
    def __len__(self):
        return self.file_names.size
    
    def __getitem__(self, idx: int):
        (name,
        h1_sfts,
        h1_timestamp,
        l1_sfts,
        l1_timestamp,
        frenquency) = read_hdf5(self.file_names[idx])
        
        target = find_target(name, self.labels)
        
        sg_h1, sg_l1 = self.preparator.fit(h1_sfts, l1_sfts)
        
        return torch.Tensor(sg_h1), torch.Tensor(sg_l1), torch.Tensor([target])
        