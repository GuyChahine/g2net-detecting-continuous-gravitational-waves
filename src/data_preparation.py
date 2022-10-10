import librosa
import numpy as np
import pandas as pd

class DataCleaning():
    
    def _min_max_scaler(self, arr: np.array, min: float, max: float):
        return (arr - min) / (max-min)

class DataTransformation():
        
    def _magphase(self, sfts):
        return librosa.magphase(sfts)
    
    def _melspectrogram(self, magnitude):
        return librosa.feature.melspectrogram(S=magnitude)
    
class MelSpectrogram_v1(DataTransformation, DataCleaning):
    
    def __init__(self, min_max_melspectrogram: pd.DataFrame):
        self.min_max_melspectrogram = min_max_melspectrogram
    
    def _get_melspectrogram(self, sfts):
        magnitude, _phase = self._magphase(sfts)
        return self._melspectrogram(magnitude)
    
    def fit(self, h1_sfts, l1_sfts):
        sg_h1 = self._get_melspectrogram(h1_sfts)
        sg_l1 = self._get_melspectrogram(l1_sfts)
        
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
        
        return sg_h1, sg_l1
        