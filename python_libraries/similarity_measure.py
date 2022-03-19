
import numpy as np
import seaborn as sns

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.stats import pearsonr

def sigmoid_5param(x, a, b, c, d, e):
    """ 5-parameter sigmoid function """
    return a / (1. + np.exp(-c * (x - d)))**e + b

def similarity_measure(method="pearson"):
    # Choose the function used to compute similarity....
    
    if method == 'euclidean':
        method = lambda x, y: euclidean_distances(x.reshape(1, -1), y.reshape(1, -1))
    elif method == 'cosine':
        method = lambda x, y: cosine_distances(x.reshape(1, -1), y.reshape(1, -1))
    elif method == 'pearson':
        method = lambda x, y: pearsonr(x, y)[0]
#    else:
#        NotImplementedError(f"The {method} method has not been implemented yet...")
    return method

def new_similarity_measure(params1, params2, d_list=np.linspace(-5, 15, num=10)):
    params2_copy = params2.copy()
    
    sim_func = similarity_measure(method="euclidean")
    
    x_axis = np.linspace(1, 45, num=50)
    curve1 = sigmoid_5param(x_axis, *params1)
    
    distances = []
    for i, d in enumerate(d_list):
        params2_copy[3] = d 
        curve2 = sigmoid_5param(x_axis, *params2_copy)
        distances.append(sim_func(curve1,curve2))
        
    return np.min(distances)

    
def correlation_matrix(df, method='pearson'):
    # Choose the function using to "similarity_measure" function
    method = similarity_measure(method=method)
    
    import inspect
    lines = inspect.getsource(method)
    print(lines)
    
    # Compute similarity between each curve
    df_corr = df.corr(method=method) 
    
    # Correct the diagonal entries. (.corr() automatically puts 1 in diagonal entries)
    for i in range(len(df_corr)):
        df_corr.iloc[i, i] = method(df.iloc[:, i].values, df.iloc[:, i].values)
    return df_corr


def plot_heatmap_correlation(ax, df_corr, kws={}):    
    # Plot Heatmap on entries
    
    sns.heatmap(df_corr, ax = ax, cmap="coolwarm_r", **kws,
                annot=True, fmt=".2f", linewidths=2.5, linecolor='w')
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, fontsize = 10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 10)


def matrix_to_scalar_min(matrix, n_assays):
	# compute the minimum distance between assays
    mask = np.triu(np.ones(n_assays) - np.eye(n_assays))
    return np.min(np.extract(mask, matrix))

def matrix_to_scalar_sum(matrix, n_assays):
	# compute the the sum of distance values between assays
    mask = np.triu(np.ones(n_assays) - np.eye(n_assays))
    return np.sum(np.extract(mask, matrix))
    