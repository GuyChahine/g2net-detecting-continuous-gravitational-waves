import sys
sys.path.insert(0, "\\".join(__file__.split("\\")[:__file__.split("\\").index("g2net-detecting-continuous-gravitational-waves")+1]))

from src.data_from_hdf5 import *

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
            infos = pd.concat([
                infos,
                pd.DataFrame({
                    "id": [d['Name']],
                    "label": [d['Label']],
                    
                    "h1_sfts_real_min": [d['H1_SFTs'].real.min()],
                    "h1_sfts_real_max": [d['H1_SFTs'].real.max()],
                    "h1_sfts_real_mean": [d['H1_SFTs'].real.mean()],
                    
                    "h1_sfts_imag_min": [d['H1_SFTs'].imag.min()],
                    "h1_sfts_imag_max": [d['H1_SFTs'].imag.max()],
                    "h1_sfts_imag_mean": [d['H1_SFTs'].imag.mean()],
                    
                    "h1_timestamp_min": [d['H1_Timestamp'].min()],
                    "h1_timestamp_max": [d['H1_Timestamp'].max()],
                    "h1_timestamp_mean": [d['H1_Timestamp'].mean()],
                    
                    "l1_sfts_real_min": [d['L1_SFTs'].real.min()],
                    "l1_sfts_real_max": [d['L1_SFTs'].real.max()],
                    "l1_sfts_real_mean": [d['L1_SFTs'].real.mean()],
                    
                    "l1_sfts_imag_min": [d['L1_SFTs'].imag.min()],
                    "l1_sfts_imag_max": [d['L1_SFTs'].imag.max()],
                    "l1_sfts_imag_mean": [d['L1_SFTs'].imag.mean()],
                    
                    "l1_timestamp_min": [d['L1_Timestamp'].min()],
                    "l1_timestamp_max": [d['L1_Timestamp'].max()],
                    "l1_timestamp_mean": [d['L1_Timestamp'].mean()],
                    
                    "frequency_min": [d['Frequency'].min()],
                    "frequency_max": [d['Frequency'].max()],
                    "frequency_mean": [d['Frequency'].mean()],
                })
            ])
    infos.set_index("id").to_csv("data/infos/train_infos.csv")


if __name__ == "__main__":
    main()