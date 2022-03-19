import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

import python_libraries.fitting_func as fitfunc
import python_libraries.utilities as utils
import python_libraries.peak_finder as peakfunc


 
def ffi_region_filter(df_in, region_width='default', sample_groups='Target'):
    """
    Filter AC curves using a region around the FFI value of the median curve.
    Data would need to be grouped by Target or CPE or AMR to calculate the median of the amplification curves.
    :param df_in: input AC dataframe
    :param region_width: range of values around FFI (e.g., region_width = 0.5 implies curves with FFIs that are not
                         within +-0.5 of the Median FFI will be filtered), if undefined it defaults to the standard
                         deviation of the FFI values.
    :param NMETA: number of columns of metadata info
    :param sample_groups: the column name used to group your data (i.e., 'Target')
    :return: Returns a filtered dataframe using the above method
    """

    df_store = []

    for group, df in df_in.groupby(sample_groups):

        median = df.iloc[:, -1].median(axis=0)  # taking the median of last column

        if region_width == 'default':
            region = df.iloc[:, -1].std()  # taking the s.d. of last column
        else:
            region = region_width

        df_filtered = df[df.iloc[:, -1].between(median - region, median + region)]
        df_store.append(df_filtered)

    df_clean_ffi = pd.concat(df_store, ignore_index=True)

    return df_clean_ffi


# 2. Filter using Constant Margin around Median
def constant_margin_filter(df_in, NMETA, margin_width=0.125, sample_groups='Target'):
    """
    Filter AC curves using an upper and lower constant margins based on the median curve.
    Curves with all points within the margins will be retained.
    :param df_in: input AC dataframe
    :param margin_width: indicate the distance between the margin and the median.
    :param NMETA: number of columns of metadata info
    :param sample_groups: the column name used to group your data (i.e., 'Target')
    :return: Returns a filtered dataframe using the above method
    """

    df_store = []
    cols = df_in.columns[NMETA:]

    for group, df in df_in.groupby(sample_groups):
        
        in_store = []
        median = df.iloc[:, NMETA:].median(axis=0)  # taking the median of each column

        for col in cols:
            in_store.append(df[col].between(median[col] - margin_width, median[col] + margin_width))

        df_in = pd.concat(in_store, axis=1)  # create a true and false df for each well (will all the cycles)
        df_filtered = df[df_in.all(axis=1)]  # will keep only the row (or wells) with True values at each cycle
        df_store.append(df_filtered)  # wells which are NOT in the median range will be discarded

    df_clean_margin = pd.concat(df_store, ignore_index=True)

    return df_clean_margin


def dtw_filter(df_in, df_mc, NMETA, sample_groups='Target'):
    """
    Filter AC curves according to their respective DTW distances from the median curve.
    :param df_in: input AC dataframe
    :param NMETA: number of columns of metadata info
    :param sample_groups: the column name used to group your data (i.e., 'Target')
    :return: Returns a filtered dataframe using the above method
    """

    df_store = []
    df_store_mc = []

    for group, df in tqdm(df_in.groupby(sample_groups)):

        median = df.iloc[:, NMETA:].median(axis=0)  # taking the median of each column
        dist_store = []

        for i in range(df.shape[0]):
            distance, _ = fastdtw(median.values, df.iloc[i, NMETA:].values, dist=euclidean)
            dist_store.append(distance)

        mc = df_mc[df_mc[sample_groups] == group]

        filter_idx = np.array(dist_store) < np.median(dist_store)  # should we keep the mean values?
        df_filtered = df[filter_idx]
        df_filtered_mc = mc[filter_idx]

        df_store.append(df_filtered)
        df_store_mc.append(df_filtered_mc)

    df_clean_dtw = pd.concat(df_store, ignore_index=True)
    df_clean_dtw_mc = pd.concat(df_store_mc, ignore_index=True)

    return df_clean_dtw, df_clean_dtw_mc, dist_store


def euclidean_filter(df_in, df_mc, NMETA, sample_groups='Target'):
    """
    Filter AC curves according to their respective euclidean distances from the median curve.
    :param df_in: input AC dataframe
    :param NMETA: number of columns of metadata info
    :param sample_groups: the column name used to group your data (i.e., 'Target')
    :return: Returns a filtered dataframe using the above method
    """

    df_store = []
    df_store_mc = []

    for group, df in tqdm(df_in.groupby(sample_groups)):

        median = df.iloc[:, NMETA:].median(axis=0) # taking the median of each column
        dist_store = []

        for i in range(df.shape[0]):
            distance = euclidean(median.values, df.iloc[i, NMETA:].values)
            dist_store.append(distance)

        mc = df_mc[df_mc[sample_groups] == group]

        filter_idx = np.array(dist_store) < np.median(dist_store)  # should we keep the mean values?
        df_filtered = df[filter_idx]
        df_filtered_mc = mc[filter_idx]

        df_store.append(df_filtered)
        df_store_mc.append(df_filtered_mc)

    df_clean_euclidean = pd.concat(df_store, ignore_index=True)
    df_clean_dtw_mc = pd.concat(df_store_mc, ignore_index=True)

    return df_clean_euclidean, df_clean_dtw_mc, dist_store


def normalize_data(df_in, NMETA):
    """
    Normalizes the curves such that they are within the range 0 to 1
    :param df_in: input AC dataframe
    :param NMETA: number of columns of metadata info
    :return: Returns dataframe containing normalized curves
    """

    df_norm = df_in.copy()

    curves = df_norm.iloc[:, NMETA:]

    curves[curves < 0] = 0
    row_min = np.array(curves.values.min(axis=1), ndmin=2).T
    row_max = np.array(curves.values.max(axis=1), ndmin=2).T

    curves_norm = (curves - row_min) / (row_max - row_min)

    df_norm.update(curves_norm)

    return df_norm


