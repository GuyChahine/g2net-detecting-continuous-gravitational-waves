import torch
import pandas as pd
import numpy as np

from src.utils import get_file_names, read_hdf5, find_target, DataCleaning

class ShortTimeFourierTransform_v1(DataCleaning):
    
    def __init__(
        self,
        root: str = "data/raw/train/",
        labels_path: str = "data/raw/train_label/_labels.csv",
        min_max_mean_shape_path: str = "data/infos/min_max_mean_shape.csv", 
    ):
        self.file_names = get_file_names(root)
        self.labels = pd.read_csv(labels_path)

        self.min_max_mean_shape = pd.read_csv(min_max_mean_shape_path)
    
    def __len__(self):
        return self.file_names.size
    
    def __getitem__(self, idx):
        (name,
        h1_sfts,
        h1_timestamp,
        l1_sfts,
        l1_timestamp,
        frequency) = read_hdf5(self.file_names[idx]).values()
        
        target = find_target(name, self.labels)
        
        h1_sfts, l1_sfts, frequency = self.fit(h1_sfts, l1_sfts, frequency)
        
        return (
            torch.Tensor(h1_sfts),
            torch.Tensor(l1_sfts),
            torch.Tensor(frequency),
            torch.Tensor([target]),
        )
    
    def _get_real_imag(self, arr: np.ndarray):
        return arr.real, arr.imag
    
    def _get_min_max(self, arr: np.ndarray):
        return np.array([arr.min(), arr.max()])
    
    def fit(self, h1_sfts, l1_sfts, frequency):
        h1_sfts = self._resizer(
            h1_sfts,
            int(self.min_max_mean_shape.h1_sfts_shape1.min())
        )
        l1_sfts = self._resizer(
            l1_sfts,
            int(self.min_max_mean_shape.l1_sfts_shape1.min())
        )
        
        h1_sfts_real, h1_sfts_imag = self._get_real_imag(h1_sfts)
        l1_sfts_real, l1_sfts_imag = self._get_real_imag(l1_sfts)
        
        h1_sfts_real = self._min_max_scaler(
            h1_sfts_real,
            float(self.min_max_mean_shape.h1_sfts_real_min.min()),
            float(self.min_max_mean_shape.h1_sfts_real_max.max()),
        )
        h1_sfts_imag = self._min_max_scaler(
            h1_sfts_imag,
            float(self.min_max_mean_shape.h1_sfts_imag_min.min()),
            float(self.min_max_mean_shape.h1_sfts_imag_max.max()),
        )
        l1_sfts_real = self._min_max_scaler(
            l1_sfts_real,
            float(self.min_max_mean_shape.l1_sfts_real_min.min()),
            float(self.min_max_mean_shape.l1_sfts_real_max.max()),
        )
        l1_sfts_imag = self._min_max_scaler(
            l1_sfts_imag,
            float(self.min_max_mean_shape.l1_sfts_imag_min.min()),
            float(self.min_max_mean_shape.l1_sfts_imag_max.max()),
        )
        
        frequency = self._min_max_scaler(
            frequency,
            float(self.min_max_mean_shape.frequency_min.min()),
            float(self.min_max_mean_shape.frequency_max.max()),
        )
        
        frequency = self._get_min_max(frequency)
        
        h1_sfts = self._concater([
            self._reshaper(h1_sfts_real, (1, *h1_sfts_real.shape)),
            self._reshaper(h1_sfts_imag, (1, *h1_sfts_imag.shape)),
        ], axis=0)
        l1_sfts = self._concater([
            self._reshaper(l1_sfts_real, (1, *l1_sfts_real.shape)),
            self._reshaper(l1_sfts_imag, (1, *l1_sfts_imag.shape)),
        ], axis=0)
        
        return h1_sfts, l1_sfts, frequency


class ShortTimeFourierTransform_v2(DataCleaning):
    
    def __init__(
        self,
        root: str = "data/raw/train/",
        labels_path: str = "data/raw/train_label/_labels.csv",
        min_max_mean_shape_path: str = "data/infos/min_max_mean_shape.csv", 
    ):
        self.file_names = get_file_names(root)
        self.labels = pd.read_csv(labels_path)

        self.min_max_mean_shape = pd.read_csv(min_max_mean_shape_path)
    
    def __len__(self):
        return self.file_names.size
    
    def __getitem__(self, idx):
        (name,
        h1_sfts,
        h1_timestamp,
        l1_sfts,
        l1_timestamp,
        _frequency) = read_hdf5(self.file_names[idx]).values()
        
        target = find_target(name, self.labels)
        
        h1_sfts_real, h1_sfts_imag, l1_sfts_real, l1_sfts_imag = self.fit(h1_sfts, l1_sfts)
        
        return (
            torch.Tensor(h1_sfts_real),
            torch.Tensor(h1_sfts_imag),
            torch.Tensor(l1_sfts_real),
            torch.Tensor(l1_sfts_imag),
            torch.Tensor([target]),
        )
    
    def _get_real_imag(self, arr: np.ndarray):
        return arr.real, arr.imag
    
    def _get_min_max(self, arr: np.ndarray):
        return np.array([arr.min(), arr.max()])
    
    def fit(self, h1_sfts, l1_sfts):
        h1_sfts = self._resizer(
            h1_sfts,
            int(self.min_max_mean_shape.h1_sfts_shape1.min())
        )
        l1_sfts = self._resizer(
            l1_sfts,
            int(self.min_max_mean_shape.l1_sfts_shape1.min())
        )
        
        h1_sfts_real, h1_sfts_imag = self._get_real_imag(h1_sfts)
        l1_sfts_real, l1_sfts_imag = self._get_real_imag(l1_sfts)
        
        h1_sfts_real = self._min_max_scaler(
            h1_sfts_real,
            float(self.min_max_mean_shape.h1_sfts_real_min.min()),
            float(self.min_max_mean_shape.h1_sfts_real_max.max()),
        )
        h1_sfts_imag = self._min_max_scaler(
            h1_sfts_imag,
            float(self.min_max_mean_shape.h1_sfts_imag_min.min()),
            float(self.min_max_mean_shape.h1_sfts_imag_max.max()),
        )
        l1_sfts_real = self._min_max_scaler(
            l1_sfts_real,
            float(self.min_max_mean_shape.l1_sfts_real_min.min()),
            float(self.min_max_mean_shape.l1_sfts_real_max.max()),
        )
        l1_sfts_imag = self._min_max_scaler(
            l1_sfts_imag,
            float(self.min_max_mean_shape.l1_sfts_imag_min.min()),
            float(self.min_max_mean_shape.l1_sfts_imag_max.max()),
        )
        
        return h1_sfts_real, h1_sfts_imag, l1_sfts_real, l1_sfts_imag


class ShortTimeFourierTransform_v3(DataCleaning):
    
    def __init__(
        self,
        root: str = "data/raw/train/",
        labels_path: str = "data/raw/train_label/_labels.csv",
        min_max_mean_shape_path: str = "data/infos/min_max_mean_shape.csv", 
    ):
        self.file_names = get_file_names(root)
        self.labels = pd.read_csv(labels_path)

        self.min_max_mean_shape = pd.read_csv(min_max_mean_shape_path)
    
    def __len__(self):
        return self.file_names.size
    
    def __getitem__(self, idx):
        (name,
        h1_sfts,
        h1_timestamp,
        l1_sfts,
        l1_timestamp,
        _frequency) = read_hdf5(self.file_names[idx]).values()
        
        target = find_target(name, self.labels)
        
        h1_sfts_real, h1_sfts_imag, l1_sfts_real, l1_sfts_imag = self.fit(h1_sfts, l1_sfts)
        
        return (
            torch.Tensor(h1_sfts_real),
            torch.Tensor(h1_sfts_imag),
            torch.Tensor(l1_sfts_real),
            torch.Tensor(l1_sfts_imag),
            torch.Tensor([target]),
        )
    
    def _get_real_imag(self, arr: np.ndarray):
        return arr.real, arr.imag
    
    def _get_min_max(self, arr: np.ndarray):
        return np.array([arr.min(), arr.max()])
    
    def fit(self, h1_sfts, l1_sfts):
        h1_sfts = self._resizer(
            h1_sfts,
            int(self.min_max_mean_shape.h1_sfts_shape1.min())
        )
        l1_sfts = self._resizer(
            l1_sfts,
            int(self.min_max_mean_shape.l1_sfts_shape1.min())
        )
        
        h1_sfts_real, h1_sfts_imag = self._get_real_imag(h1_sfts)
        l1_sfts_real, l1_sfts_imag = self._get_real_imag(l1_sfts)
        
        h1_sfts_real = self._min_max_scaler(
            h1_sfts_real,
            float(self.min_max_mean_shape.h1_sfts_real_min.min()),
            float(self.min_max_mean_shape.h1_sfts_real_max.max()),
        )
        h1_sfts_imag = self._min_max_scaler(
            h1_sfts_imag,
            float(self.min_max_mean_shape.h1_sfts_imag_min.min()),
            float(self.min_max_mean_shape.h1_sfts_imag_max.max()),
        )
        l1_sfts_real = self._min_max_scaler(
            l1_sfts_real,
            float(self.min_max_mean_shape.l1_sfts_real_min.min()),
            float(self.min_max_mean_shape.l1_sfts_real_max.max()),
        )
        l1_sfts_imag = self._min_max_scaler(
            l1_sfts_imag,
            float(self.min_max_mean_shape.l1_sfts_imag_min.min()),
            float(self.min_max_mean_shape.l1_sfts_imag_max.max()),
        )
        
        h1_sfts_real = self._reshaper(h1_sfts_real, (1, *h1_sfts_real.shape))
        h1_sfts_imag = self._reshaper(h1_sfts_imag, (1, *h1_sfts_imag.shape))
        l1_sfts_real = self._reshaper(l1_sfts_real, (1, *l1_sfts_real.shape))
        l1_sfts_imag = self._reshaper(l1_sfts_imag, (1, *l1_sfts_imag.shape))
        
        return h1_sfts_real, h1_sfts_imag, l1_sfts_real, l1_sfts_imag