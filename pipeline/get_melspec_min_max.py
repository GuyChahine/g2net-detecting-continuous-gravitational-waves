import pandas as pd
import json

import sys
sys.path.insert(0, "\\".join(__file__.split("\\")[:__file__.split("\\").index("g2net-detecting-continuous-gravitational-waves")+1]))

from src.mel_spectrogram import MelSpectrogram_v1
from src.classic_data import batch_generator
from src.utils import get_file_names

def get_min_man_melpec(sfts):
    dt = MelSpectrogram_v1()
    magnitude, phase = dt._magphase(sfts)
    mel_spec = dt._melspectrogram(magnitude)
    return mel_spec.min(), mel_spec.max(), mel_spec.shape[1]

def main():
    BATCH_SIZE = 32
    DATASET_PATH = "data/raw/train/"
    LABEL_PATH = "data/raw/train_label/_labels.csv"
    
    generator = batch_generator(BATCH_SIZE, get_file_names(DATASET_PATH), pd.read_csv(LABEL_PATH))
    
    h1_min_melspec, h1_max_melspec, h1_shape_melspec = [], [], []
    l1_min_melspec, l1_max_melspec, l1_shape_melspec = [], [], []
    for i_batch, data in enumerate(generator):
        print(f"BATCH_NB: {i_batch+1}", end="\r")
        for d in data:
            min_max_h1 = get_min_man_melpec(d['H1_SFTs'])
            h1_min_melspec.append(min_max_h1[0])
            h1_max_melspec.append(min_max_h1[1])
            h1_shape_melspec.append(min_max_h1[2])
            min_max_l1 = get_min_man_melpec(d['L1_SFTs'])
            l1_min_melspec.append(min_max_l1[0])
            l1_max_melspec.append(min_max_l1[1])
            l1_shape_melspec.append(min_max_l1[2])
            
        pd.DataFrame(dict(
            h1_min_melspec=[min(h1_min_melspec)],
            h1_max_melspec=[max(h1_max_melspec)],
            h1_shape_melspec=[min(h1_shape_melspec)],
            l1_min_melspec=[min(l1_min_melspec)],
            l1_max_melspec=[max(l1_max_melspec)],
            l1_shape_melspec=[min(l1_shape_melspec)],
        )).to_csv("data/infos/min_max_melspectrogram.csv")

if __name__ == "__main__":
    main()