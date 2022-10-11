import librosa
import torch
import numpy as np
import pandas as pd

from src.utils import get_file_names, read_hdf5, find_target

def dataset_split(dataset: torch.utils.data.Dataset, valid_size: float):
    train_length = int(len(dataset) * (1 - valid_size))
    valid_length = int(len(dataset) * valid_size)
    return torch.utils.data.random_split(dataset, [train_length, valid_length])

class DataCleaning():
    
    def _min_max_scaler(self, arr: np.ndarray, min: float, max: float):
        return (arr - min) / (max-min)

    def _resizer(self, arr: np.ndarray, min_shape: int):
        return np.delete(arr, slice(min_shape, arr.shape[1]), axis=1)
    
    def _reshaper(self, arr: np.ndarray, new_shape: tuple):
        return arr.reshape(new_shape)

class DataTransformation():
        
    def _magphase(self, sfts):
        return librosa.magphase(sfts)
    
    def _melspectrogram(self, magnitude):
        return librosa.feature.melspectrogram(S=magnitude)
    
class MelSpectrogram_v1(DataTransformation, DataCleaning):
    
    def __init__(
        self,
        root: str = "data/raw/train/",
        labels_path: str = "data/raw/train_label/_labels.csv",
        min_max_sg_path: str = "data/infos/min_max_melspectrogram.csv",
    ):
        self.file_names = get_file_names(root)
        self.labels = pd.read_csv(labels_path)
        
        self.min_max_melspectrogram = pd.read_csv(min_max_sg_path)
    
    def __len__(self):
        return self.file_names.size
    
    def __getitem__(self, idx: int):
        (name,
        h1_sfts,
        _h1_timestamp,
        l1_sfts,
        _l1_timestamp,
        _frenquency) = read_hdf5(self.file_names[idx]).values()
        
        target = find_target(name, self.labels)
        
        sg_h1, sg_l1 = self.fit(h1_sfts, l1_sfts)
        
        return torch.Tensor(sg_h1), torch.Tensor(sg_l1), torch.Tensor([target])
    
    def _get_melspectrogram(self, sfts):
        magnitude, _phase = self._magphase(sfts)
        return self._melspectrogram(magnitude)
    
    def fit(self, h1_sfts, l1_sfts):
        sg_h1 = self._resizer(h1_sfts, int(self.min_max_melspectrogram.h1_shape_melspec))
        sg_l1 = self._resizer(l1_sfts, int(self.min_max_melspectrogram.l1_shape_melspec))
        
        sg_h1 = self._get_melspectrogram(sg_h1)
        sg_l1 = self._get_melspectrogram(sg_l1)
        
        sg_h1 = self._min_max_scaler(
            sg_h1,
            float(self.min_max_melspectrogram.h1_min_melspec),
            float(self.min_max_melspectrogram.h1_max_melspec)
        )
        sg_l1 = self._min_max_scaler(
            sg_l1,
            float(self.min_max_melspectrogram.l1_min_melspec),
            float(self.min_max_melspectrogram.l1_max_melspec)
        )
        
        sg_h1 = self._reshaper(sg_h1, (1, *sg_h1.shape))
        sg_l1 = self._reshaper(sg_l1, (1, *sg_l1.shape))
        
        return sg_h1, sg_l1
        
class ShotTimeFourierTransform_v1():
    
    def __init__(self):
        pass
    
    def fit(self):
        pass