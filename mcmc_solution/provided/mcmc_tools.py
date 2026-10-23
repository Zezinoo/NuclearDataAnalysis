import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde

import pandas as pd
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica", 
    "font.size":14})


# compute poisson likelihood between a model array and histogrammed data
def compute_poisson_loglikelihood(model, histo_data):
    logf = np.log(sp.special.factorial(histo_data))
    logf[np.isinf(logf)==True] = ((histo_data[np.isinf(logf)==True])*
                                  np.log(histo_data[np.isinf(logf)==True])-
                                  histo_data[np.isinf(logf)==True]+1.)
    LL = np.sum(-model+histo_data*np.log(model)-logf)
    return LL
    
# compute normal likelihood between a model array and histogrammed data
def compute_normal_loglikelihood(model, histo_data, sigma):
    LL = np.sum(-(model-histo_data)**2/(2.*sigma**2))
    return LL

#function to draw the posterior distribution
def pairplot_from_array(parameter, column_names=None, hue=None, save_path=None,bins=20):
    if not isinstance(parameter, np.ndarray):
        raise ValueError("parameter is not a numpy array")

    if column_names is None:
        column_names = [f"Columns_{i}" for i in range(parameter.shape[1])]

    df = pd.DataFrame(parameter, columns=column_names)

    if hue is not None and hue not in column_names:
        raise ValueError(f"The columns '{hue}' ")
    
    plot_kws = {"bins": bins}

    
    if len(parameter)<=10000:
        sns.pairplot(df, hue=hue, corner=True, kind="kde", plot_kws=plot_kws)
    else:
        sns.pairplot(df, hue=hue, corner=True, kind="hist",diag_kind="kde", plot_kws=plot_kws)

    if save_path!=None:
        plt.savefig(save_path)
    #plt.show()
    
def plot_convergence(parameter, column_names=None):
    if not isinstance(parameter, np.ndarray):
        raise ValueError("parameter is not a numpy array")

    if column_names is None:
        column_names = [f"Columns_{i}" for i in range(parameter.shape[1])]
    for i in range(parameter.shape[1]):
        fig, ax = plt.subplots(1,1,figsize=(8, 5))
        ax.plot(parameter[:,i])
        ax.set_xlabel("step")
        ax.set_ylabel(column_names[i])
        ax.set_xscale("log")
    #plt.show()

    
def plot_convergence_method(parameter, column_names=None, method_names=None):
    if not isinstance(parameter, np.ndarray):
        raise ValueError("parameter is not a numpy array")

    if column_names is None:
        column_names = [f"Columns_{i}" for i in range(parameter.shape[2])]
    if method_names is None:
        method_names = [f"method_{i}" for i in range(parameter.shape[0])]
    for i in range(parameter.shape[2]):
        fig, ax = plt.subplots(1,1,figsize=(8, 5))
        for j in range(parameter.shape[0]):
            ax.plot(parameter[j,:,i],label=method_names[j],alpha=0.5)
        ax.set_xlabel("step")
        ax.legend()
        ax.set_ylabel(column_names[i])
        ax.set_xscale("log")
    #plt.show()

def autocorrelation(data):
    n = len(data)
    
    mean = np.mean(data)
    centered_data = data - mean
    
    variance = np.dot(centered_data, centered_data)
    
    result = np.correlate(centered_data, centered_data, mode='full') / variance
    result = result[n-1:]
    return result    

def plot_autocorrelation(parameter, column_names=None):
    if not isinstance(parameter, np.ndarray):
        raise ValueError("parameter is not a numpy array")

    if column_names is None:
        column_names = [f"Columns_{i}" for i in range(parameter.shape[1])]
    for i in range(parameter.shape[1]):
        fig, ax = plt.subplots(1,1,figsize=(8, 5))
        ax.plot(autocorrelation(parameter[:,i]))
        ax.set_xlabel("lag")
        ax.set_ylabel("autocorrelation("+column_names[i]+")")
        ax.set_xscale("log")
    #plt.show()

    
def plot_autocorrelation_method(parameter, column_names=None, method_names=None):
    if not isinstance(parameter, np.ndarray):
        raise ValueError("parameter is not a numpy array")

    if column_names is None:
        column_names = [f"Columns_{i}" for i in range(parameter.shape[2])]
    if method_names is None:
        method_names = [f"method_{i}" for i in range(parameter.shape[0])]
    for i in range(parameter.shape[2]):
        fig, ax = plt.subplots(1,1,figsize=(8, 5))
        for j in range(parameter.shape[0]):
            ax.plot(autocorrelation(parameter[j,:,i]),
                    label=method_names[j],alpha=0.5)
        ax.set_xlabel("lag")
        ax.legend()
        ax.set_ylabel("autocorrelation("+column_names[i]+")")
        ax.set_xscale("log")
    #plt.show()

def plot_hist(hist,xbin, title = None):
    fig, ax = plt.subplots(1,1,figsize=(8, 5))
    ax.hist((xbin[:-1]+xbin[1:])/2.,bins=xbin,weights=hist)
    ax.set_xlabel("x_bin")
    if title != None:
        ax.set_title(title)
    #plt.show()
    
def plot_acceptance_rate(parameters, title = None, number_of_batch = 10):
    n = len(parameters)
    batch_size = int(n/number_of_batch)
    print(batch_size)
    acceptation = []
    for i in range(number_of_batch):
        unique_values, occurrences = np.unique(parameters[i*batch_size:(i+1)*batch_size,:], 
                                               axis=0, return_counts=True)
        acceptation.append(float(len(unique_values))/float(batch_size))
    fig, ax = plt.subplots(1,1,figsize=(8, 5))
    ax.plot(acceptation,color="red")
    ax.set_xlabel("batch")
    ax.set_ylabel("acceptance ratio")
    if title != None:
        ax.set_title(title)
    #plt.show()

################################################################################
def get_credible_interval(data, level=0.68):
    low = np.percentile(data, (1 - level) / 2 * 100)
    high = np.percentile(data, (1 + level) / 2 * 100)
    median = np.median(data)
    return low, median, high


def compute_levels(Z, levels=[0.68, 0.95]):
    sorted_Z = np.sort(Z.ravel())[::-1]
    cumsum_Z = np.cumsum(sorted_Z)
    cumsum_Z /= cumsum_Z[-1]
    levels_values = []
    for lev in levels:
        idx = np.searchsorted(cumsum_Z, lev)
        levels_values.append(sorted_Z[idx])
    return levels_values


def get_credible_interval(data, level=0.68):
    low = np.percentile(data, (1 - level) / 2 * 100)
    high = np.percentile(data, (1 + level) / 2 * 100)
    median = np.median(data)
    return low, median, high

    fig, axes = plt.subplots(n_params, n_params, figsize=(2.5 * n_params, 2.5 * n_params))




def plot_contours( axes, df, labels, color, title ):
    n_params = len(labels)
    credible_stats = [get_credible_interval(df[labels[i]]) for i in range(n_params)]
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i == j:
                data = df[labels[i]]
                
                low, median, high = credible_stats[i]

                

                kde = gaussian_kde(data)
                x_grid = np.linspace(data.min(), data.max(), 1000)
                ax.plot(x_grid, kde(x_grid), color=color,label=title)
                #ax.legend();
                ax.axvline(median, color=color, linestyle='-')
                ax.axvline(low, color=color, linestyle='--')
                ax.axvline(high, color=color, linestyle='--')

            elif i > j:
                # En dessous : densité 2D KDE avec contours crédibles
                x = df[labels[j]]
                y = df[labels[i]]
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy)

                x_grid = np.linspace(x.min(), x.max(), 100)
                y_grid = np.linspace(y.min(), y.max(), 100)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                x_low, x_med, x_high = credible_stats[j]
                y_low, y_med, y_high = credible_stats[i]

                levels_values = compute_levels(Z, levels=[0.68, 0.95,0.995])
                ax.contour(X, Y, Z, levels=levels_values[::-1], colors=color, linewidths=1.5)
                x_low, x_med, x_high = credible_stats[j]
                y_low, y_med, y_high = credible_stats[i]
                ax.axvline(x_med, color=color, linestyle='-')
                ax.axhline(y_med, color=color, linestyle='-')

            else:
                ax.axis('off')

            # Labels
            if i == n_params - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticks([])
            if (j == 0) and (i != 0):
                ax.set_ylabel(labels[i])
            else:
                ax.set_yticks([])
