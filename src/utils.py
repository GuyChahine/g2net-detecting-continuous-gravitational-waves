import os
import h5py
import numpy as np
import pandas as pd
import torch

def get_file_names(root: str):
    for _, _, files in os.walk(root):
        return np.array([root + file_name for file_name in files])
    
def read_hdf5(path: str):
    file = h5py.File(path)
    return {
        "Name": list(file.keys())[0],
        "H1_SFTs": np.array(file[list(file.keys())[0]]["H1"]["SFTs"]),
        "H1_Timestamp": np.array(file[list(file.keys())[0]]["H1"]["timestamps_GPS"]),
        "L1_SFTs": np.array(file[list(file.keys())[0]]["L1"]["SFTs"]),
        "L1_Timestamp": np.array(file[list(file.keys())[0]]["L1"]["timestamps_GPS"]),
        "Frequency": np.array(file[list(file.keys())[0]]["frequency_Hz"]),
    }
    
def find_target(name: str, labels: pd.DataFrame):
    return float(labels[labels.id == name].target)

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