import os
import h5py
import numpy as np
import pandas as pd

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
    return int(labels[labels.id == name].target)