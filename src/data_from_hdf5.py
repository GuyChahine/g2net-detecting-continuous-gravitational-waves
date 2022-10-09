import h5py
import pandas as pd
import numpy as np
import os

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

def batch_generator(batch_size: int, file_names: list, labels: pd.DataFrame):
    batch_names = np.split(file_names, np.arange(batch_size, file_names.size, batch_size))
    for names in batch_names:
        batch_data = []
        for name in names:
            data = read_hdf5(name)
            data["Label"] = labels[labels.id == data['Name']]['target'].values[0]
            batch_data.append(data)
        yield batch_data
