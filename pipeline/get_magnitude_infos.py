import sys
sys.path.insert(0, "\\".join(__file__.split("\\")[:__file__.split("\\").index("g2net-detecting-continuous-gravitational-waves")+1]))

import pandas as pd
from librosa import magphase

from src.utils import get_file_names
from src.classic_data import batch_generator

def main():
    BATCH_SIZE = 128
    file_names = get_file_names("data/raw/train/")
    data_generator = batch_generator(
        batch_size=BATCH_SIZE,
        file_names=file_names,
        labels=pd.read_csv("data/raw/train_label/_labels.csv"),
    )
    infos = pd.DataFrame()
    for i_batch, data in enumerate(data_generator):
        for i, d in enumerate(data):
            print(f"Batch: {i_batch+1}/{int((len(file_names)/BATCH_SIZE)+1)} | Data: {i+1}/{len(data)}", end="\r")
            
            h1_magnitude, _h1_phase = magphase(d['H1_SFTs'])
            l1_magnitude, _l1_phase = magphase(d['L1_SFTs'])
            
            infos = pd.concat([
                infos,
                pd.DataFrame({
                    "id": [d['Name']],
                    "label": [d['Label']],
                    
                    "h1_magnitude_min": [h1_magnitude.min()],
                    "h1_magnitude_max": [h1_magnitude.max()],
                    "h1_magnitude__mean": [h1_magnitude.mean()],
                    
                    "h1_magnitude_shape0": [h1_magnitude.shape[0]],
                    "h1_magnitude_shape1": [h1_magnitude.shape[1]],
                    
                    "h1_timestamp_min": [d['H1_Timestamp'].min()],
                    "h1_timestamp_max": [d['H1_Timestamp'].max()],
                    "h1_timestamp_mean": [d['H1_Timestamp'].mean()],
                    
                    "l1_magnitude_min": [l1_magnitude.min()],
                    "l1_magnitude_max": [l1_magnitude.max()],
                    "l1_magnitude_mean": [l1_magnitude.mean()],
                    
                    "l1_magnitude_shape0": [l1_magnitude.shape[0]],
                    "l1_magnitude_shape1": [l1_magnitude.shape[1]],
                    
                    "l1_timestamp_min": [d['L1_Timestamp'].min()],
                    "l1_timestamp_max": [d['L1_Timestamp'].max()],
                    "l1_timestamp_mean": [d['L1_Timestamp'].mean()],
                    
                    "frequency_min": [d['Frequency'].min()],
                    "frequency_max": [d['Frequency'].max()],
                    "frequency_mean": [d['Frequency'].mean()],
                })
            ])
    infos.set_index("id").to_csv("data/infos/magnitude_infos.csv")


if __name__ == "__main__":
    main()