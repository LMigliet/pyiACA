
import pandas as pd 
import numpy as np
from scipy import signal 
import peakutils

from tqdm.auto import tqdm
import time

def find_peaks(df, height=0.01, width=4):
    data = df.values
    x = df.index.values
    x_lower = x.min()

    data = data - np.median(data, axis=0)

    peaks = []
    for i in tqdm(range(data.shape[1])):
        peakind, _ = signal.find_peaks(data[:, i], height=height)
        if len(peakind):
                peaks_x = peakutils.interpolate(x, data[:, i], 
                                                ind=peakind, width=width)
                peaks.append( peaks_x[peaks_x>x_lower] )
#             peaks.append( x[peakind] )
        else:
            peaks.append([])
            
    return peaks