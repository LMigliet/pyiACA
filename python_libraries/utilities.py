import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
import matplotlib.ticker as ticker

from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpl_toolkits.axes_grid1 import make_axes_locatable


def my_heatmap(cm, ax, cmap=plt.cm.Blues):
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=7.0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax)

    fmt = '.1f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="black" if cm[i, j] < 2 else "white")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")


def colourblind():
	import matplotlib as mpl
	from cycler import cycler
	from matplotlib.colors import to_hex
	mpl.rcParams['axes.prop_cycle'] = cycler(color=[to_hex(i) for i in [
		(0,0.45,0.70), 
		(0.9, 0.6, 0.0), 
		(0.0, 0.60, 0.50), 
		(0.8, 0.4, 0), 
		(0.35, 0.7, 0.9), 
		(0.8, 0.6, 0.7), 
		(0,0,0), 
		(0.5, 0.5, 0.5), 
		(0.95, 0.9, 0.25)]])


def compute_cts(df, thresh):
    """
    Function to compute Ct for amplification curves.
        df (pd.DataFrame) - dataframe where each column is an amplification curve
        thresh (float) - threshold for computing Ct
    """
    n_rows, n_columns = df.shape

    # Extract insides of df, so as to work with numpy only
    # Note: This is because we will make this function numpy compatible in the future
    
    cols = df.columns
    x = df.index
    df = df.values

    cts = np.zeros(n_columns)

    for i in range(n_columns):
        y = df[:, i]
        idx, = np.where(y > thresh)
        if len(idx) > 0:
            idx = idx[0]
            p1 = y[idx-1]
            p2 = y[idx]
            t1 = x[idx-1]
            t2 = x[idx]
            cts[i] = t1 + (thresh - p1)*(t2 - t1)/(p2 - p1)
        else:
            cts[i] = -99

    return pd.DataFrame({'Ct': cts}, index=cols)


def order_columns(df, NMETA):
    # sorting the column indexes. This is valid for both AC or MC.
    unordered_idxs = [float(i) for i in df.columns[NMETA:]]
    df.columns = list(df.columns[:NMETA]) + unordered_idxs
    df = df[list(df.columns[:NMETA]) + sorted(unordered_idxs)]
    return df


def order_temperatures(df,N):
    unordered_temps = [float(i) for i in df.columns[N:]]
    df.columns = list(df.columns[:N]) + unordered_temps
    df = df[list(df.columns[:N]) + sorted(unordered_temps)]
    return df


def interpolate_melting(df, desired_index=np.linspace(65.5, 97, num=int((97-65.5)/0.1)+1)):
    df_interpolated = df.transpose()
    
    # Safety: convert temperatures to float
    df_interpolated.index = df_interpolated.index.astype(float)
    df_interpolated = df_interpolated.reindex(df_interpolated.index.union(desired_index))
    df_interpolated = df_interpolated.sort_index().interpolate()
    df_interpolated = df_interpolated.loc[desired_index,:]
    
    df_interpolated = df_interpolated.transpose()
    
    return df_interpolated


def tranpose_apply_tranpose(df, func):
    # transpose df, apply function and then transpose again
    return func(df.transpose()).transpose()


def apply_func_to_columns(df, N, func):
    """ applies a function to df after N columns"""
    return df.iloc[:,:N].join(func(df.iloc[:, N:]))


def apply_processing(df, func, N):
    func_wrapper = lambda x: tranpose_apply_tranpose(x, func=func)
    return apply_func_to_columns(df, N=N, func=func_wrapper)
 

def concatenate_dfs(mydict, key):
    """ Function to concatenate list of dataframes (from dictionaries)"""
    return pd.concat([v[key] for k, v in mydict.items()], axis=1)   


def df_down_selection(df, col_index, sample_list):
    """
    function to filter the dataframe. It removes the samples which you 
    are not considering in your analysis
    df: pandas datafram
    col_index: the df column where the loc will be applied
    sample_list: a list of samples to exclude
    - return a dataframe 
    """
    for sample in sample_list:
        df = df[df[col_index]!=sample]
    return df


