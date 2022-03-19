
import pandas as pd 
import numpy as np

import scipy.optimize as opt
from multiprocessing import Pool
import numpy.polynomial.polynomial as poly

from tqdm.auto import tqdm
import time
import itertools


def sigmoid_5param(x, a, b, c, d, e):
    """ 5-parameter sigmoid function """
    return a / (1. + np.exp(-c * (x - d)))**e + b


def my_fit(series, f, p0, maxfev, bounds):
    """ Fit a function a pd.Series """
    x = series.index.values
    y = series.values
    sig_params, _ = opt.curve_fit(f, x, y, p0=p0, maxfev=maxfev, bounds=bounds)
    return sig_params


def fit_sigmoid(raw_curves, fitting_kwargs={}):
    """
    Function to perform sigmoidal fitting
    
    raw_curves (pd.DataFrame) - dataframe where each column is an amplification curve
    fitting_kwargs(dict) - dictionary with fitting parameters for sigmoidal fitting
    
    """
    
    # Set default values...
    f = fitting_kwargs.get('f', 'sigmoid4')
    p0 = fitting_kwargs.get('p0', (0.,0.,0.,0.))
    parallel = fitting_kwargs.get('parallel', False)
    n_cores = fitting_kwargs.get('n_cores', 1)
    maxfev = fitting_kwargs.get('maxfev', 10000)
    bounds = fitting_kwargs.get('bounds', (-100, 100))
    verbose = fitting_kwargs.get('verbose', False)
    progress_bar = fitting_kwargs.get('progress_bar', False)

    start_time = time.time()
        
    # Pick out function definition
    if f == 'sigmoid4':
        func = sigmoid_4param
    elif f == 'sigmoid5':
        func = sigmoid_5param
    else:
        raise NotImplementedError('{} is not implemented...'.format(f))

    # If parallel option is chosen...
    if parallel:
        # Split dataframe into n_cores 
        df_split = np.array_split(raw_curves, n_cores, axis=1)
        
        # Create n_cores workers
        pool = Pool(n_cores)
        
        # Send task to each worker
        sigmoid_params = pd.concat(pool.map(my_fit_wrapper, 
                                                 zip(df_split, 
                                                     itertools.repeat(func), 
                                                     itertools.repeat(p0),
                                                     itertools.repeat(maxfev),
                                                     itertools.repeat(bounds),
                                                     itertools.repeat(progress_bar)
                                                    )), axis=1)
        # Close workers
        pool.close()
        pool.join() # This waits for all workers to terminate...
    else:
        if progress_bar:
            tqdm.pandas()
            sigmoid_params = raw_curves.progress_apply(lambda x: my_fit(x, func, p0, maxfev, bounds), axis=0)
        else:
            sigmoid_params = raw_curves.apply(lambda x: my_fit(x, func, p0, maxfev, bounds), axis=0)

    sigmoid_curves = sigmoid_params.apply(lambda p: func(raw_curves.index, *p))
    sigmoid_curves.index = raw_curves.index 
    end_time = time.time()

    if verbose:
        print('Elapsed time: {} seconds'.format(end_time - start_time))
    
    return func, sigmoid_params, sigmoid_curves


def correlation_sigm(a,b):
    """
    will look at the difference of each value of the curves (makes nega into positive). 
    this is the area between the curves.
    """
    return np.sqrt(((a-b)**2).sum()) 


def normalize_by_sigmoid(df, sig_params):
    return df / sig_params.loc[0]


def normalize_on_background(df):
    return (df / df.iloc[0, :]) - 1


def normalize_by_sigmoid_wrapper(df, fitting_kwargs):
    _, sigmoid_params, sigmoids = fit_sigmoid(df, fitting_kwargs=fitting_kwargs)
    norm_simgoids = normalize_by_sigmoid(df, sigmoid_params)
    return sigmoid_params, sigmoids, norm_simgoids


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
            print(idx)
            cts[i] = t1 + (thresh - p1)*(t2 - t1)/(p2 - p1)
        else:
            cts[i] = -99

    return pd.DataFrame({'Ct': cts}, index=cols)


def remove_background(df, order=0, n_ct_fit=5, n_ct_skip=0):
    """
    Function to remove background from amplification curve using polynomial fit.
        df (pd.DataFrame) - dataframe where each column is an amplification curve
        order (int) - order of polynomial fit
        n_ct_fit (int) - number of samples to use for fit. This is the number of CT you will use
        n_ct_skip (int) - number of initial samples to skip. In this funct is n of CTs to skip before fitting
            (i.e. if n=5 and n_skip=2, use 3rd to 8th data point for fitting)
    """

    # If order < 0, do nothing... Maybe raise ValueError in the future?
    if order < 0:
        return df

    n_rows, n_columns = df.shape

    # x --> indices to use for fitting
    x = np.arange(n_ct_skip, n_ct_skip+n_ct_fit)

    # x_new --> indices for extrapolating the background fit
    x_new = np.arange(n_rows)

    # If order == 0, simply remove mean
    if order == 0:
        return df- df.iloc[n_ct_skip:n_ct_skip+n_ct_fit, :].mean()
    # Otherwise, fit a polynomial
    else:
        df_new = df.copy()
        for col in tqdm(range(n_columns)):
            coefs = poly.polyfit(x, df.iloc[x, col], order)
            df_new.iloc[:, col] = poly.polyval(x_new, coefs)

        return df - df_new

    
def my_fit_wrapper(tup):
    """ Wrapper used for multiprocessing sigmoidal fitting. Note: Needs single argument. """
    df, f, p0, maxfev, bounds, progress_bar = tup
    if progress_bar:
        tqdm.pandas()
        return df.progress_apply(lambda x: my_fit(x, f, p0, maxfev, bounds), axis=0)
    else:
        return df.apply(lambda x: my_fit(x, f, p0, maxfev, bounds), axis=0)
    
def extract_curves(filename):
    """
    It takes file of raw data of the amplification data (AC) from dPCR
    and reorganise them them into a dataframe
    :param filename: AC file from dPCR
    :return: df with "cycle" as index(row) and "well" as column. Each cell has raw fluo value.
    """
    df_ac = pd.read_csv(filename, sep="\t", header=None)
    df_ac.index = df_ac.iloc[:, 0]
    df_ac = df_ac.iloc[:, 1::2]
    df_ac.columns = ['Well ' + str(i) for i in range(1, len(df_ac.columns)+1)]
    df_ac.index.name = 'Cycle'
    return df_ac    


def is_positive_iloc(df, ampl_thresh, ct_thresh):
    return (df.iloc[ct_thresh:, :] > ampl_thresh).all(axis='rows')


def is_positive_loc(df, ampl_thresh, ct_thresh):
    return (df.loc[ct_thresh:, :] > ampl_thresh).all(axis='rows')
