import torch
import pandas as pd
import numpy as np
from librosa import magphase

from src.utils import get_file_names, read_hdf5, find_target, DataCleaning

class Magnitude_v1(DataCleaning):
    
    def __init__(
        self,
        root: str = "data/raw/train/",
        labels_path: str = "data/raw/train_label/_labels.csv",
        magnitude_infos_path: str = "data/infos/magnitude_infos.csv", 
    ):
        self.file_names = get_file_names(root)
        self.labels = pd.read_csv(labels_path)

        self.magnitude_infos = pd.read_csv(magnitude_infos_path)
    
    def __len__(self):
        return self.file_names.size
    
    def __getitem__(self, idx):
        (name,
        h1_sfts,
        _h1_timestamp,
        l1_sfts,
        _l1_timestamp,
        _frequency) = read_hdf5(self.file_names[idx]).values()
        
        target = find_target(name, self.labels)
        
        h1_magnitude, l1_magnitude = self.fit(h1_sfts, l1_sfts)
        
        return (
            torch.Tensor(h1_magnitude),
            torch.Tensor(l1_magnitude),
            torch.Tensor([target]),
        )
    
    def _get_magnitude(self, arr: np.ndarray):
        return magphase(arr)[0]
    
    def _get_min_max(self, arr: np.ndarray):
        return np.array([arr.min(), arr.max()])
    
    def fit(self, h1_sfts, l1_sfts):
        h1_magnitude = self._get_magnitude(h1_sfts)
        l1_magnitude = self._get_magnitude(l1_sfts)
        
        h1_magnitude = self._resizer(
            h1_magnitude,
            int(self.magnitude_infos.h1_magnitude_shape1.min())
        )
        l1_magnitude = self._resizer(
            l1_magnitude,
            int(self.magnitude_infos.l1_magnitude_shape1.min())
        )
        
        h1_magnitude = self._min_max_scaler(
            h1_magnitude,
            float(self.magnitude_infos.h1_magnitude_min.min()),
            float(self.magnitude_infos.h1_magnitude_max.max()),
        )
        l1_magnitude = self._min_max_scaler(
            l1_magnitude,
            float(self.magnitude_infos.l1_magnitude_min.min()),
            float(self.magnitude_infos.l1_magnitude_max.max()),
        )
        
        h1_magnitude = self._reshaper(h1_magnitude, (1, *h1_magnitude.shape))
        l1_magnitude = self._reshaper(l1_magnitude, (1, *l1_magnitude.shape))
        
        return h1_magnitude, l1_magnitude