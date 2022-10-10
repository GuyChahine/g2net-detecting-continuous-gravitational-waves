import numpy as np
import pandas as pd

from utils import read_hdf5

def batch_generator(batch_size: int, file_names: list, labels: pd.DataFrame):
    batch_names = np.split(file_names, np.arange(batch_size, file_names.size, batch_size))
    for names in batch_names:
        batch_data = []
        for name in names:
            data = read_hdf5(name)
            data["Label"] = labels[labels.id == data['Name']]['target'].values[0]
            batch_data.append(data)
        yield batch_data
