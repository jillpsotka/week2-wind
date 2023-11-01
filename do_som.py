from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from som_class import SOM, BMUs, BMU_frequency, colourmap_2D
import xarray as xr
import time


def prep_gefs(ds, step_int=0):
    ds_arr = ds.gh.to_numpy()
    ds_arr = np.reshape(ds_arr,(ds.time.shape[0],ds.latitude.shape[0]*ds.longitude.shape[0]))


    return ds_arr



def prep_obs(ds, gefs, step_int):
 
    ds = ds.where(~np.isnan(ds.Wind),drop=True)  # get rid of nan obs

    times = gefs.time + gefs.step.values  # actual valid time of forecast
    indices = times.isin(ds.index)  # indices of valid times that have obs
    

    times_new = times.where(indices, drop=True)  # valid times that have obs

    gefs = gefs.sel(time = (times_new - gefs.step.values))  # get rid of gefs times that don't have obs
    ds = ds.sel(index=times_new).Wind.to_numpy()  # get rid of obs that aren't in gefs


    return ds, gefs



def pca_gefs(gefs):
    # do PCA to see the ratio of nodes in the ideal map
    pca = PCA(n_components=10)
    PCs = pca.fit_transform(gefs)
    frac_var = pca.explained_variance_ratio_
    var = pca.explained_variance_
    std = var ** 0.5
    eigvecs = pca.components_


    return None



def train_som(gefs_arr, obs_arr):
    # normalize data (this is actually the z score, could try other methods of standardization)
    gefs_arr = (gefs_arr - np.mean(gefs_arr,axis=0)) / np.std(gefs_arr,axis=0)  # axis 1 is the space axis
    obs_arr = (obs_arr - np.mean(obs_arr)) / np.std(obs_arr)

    #pca_gefs(gefs_arr)


    N_nodes = Nx*Ny
    learning_rate = 1e-2
    N_epochs = 50
    colours_list = 'default2'

    som = SOM(Nx, Ny, gefs_arr, N_epochs, linewidth=4, colours_list=colours_list)
    som.initialize_map(node_shape='hex')

    # train
    tic = time.perf_counter()
    som.train_map(learning_rate)
    toc = time.perf_counter()
    print(f'Finished training map in {toc - tic:0.2f} seconds')

    z = som.z  # pattern of each node
    indices = np.arange(N_nodes).reshape(Nx,Ny).T.flatten()
    bmus = BMUs(som)  # the nodes that each forecast belongs to -> use this to look at wind
    freq = BMU_frequency(som)  # frequency of each node
    QE = som.QE()  # quantization error
    TE = som.TE()  # topographic error


    return z, indices, bmus


def plot_som(Nx, Ny, z, indices):
    fig, axes = plt.subplots(nrows=Ny, ncols=Nx)
    i = 0
    k = 3645
    for kk, ax in enumerate(axes.flatten()):
        var = z[indices[kk],i:k].reshape(45,81)
        ax.contourf(lon, lat, var, cmap='RdBu_r')
    plt.show()

    #plt.plot(range(gefs_arr.shape[0]), bmus, 'bo--')


def wind_distributions(bmus):
    N_nodes = 30
    distributions = np.empty(N_nodes)
    for i in range(N_nodes):
        distributions[i] = obs[np.where(bmus==i+1)[0]]  # wind obs that belong to this node -> this is our distribution

    return distributions


if __name__ ==  "__main__":
    step = 1
    lat = np.arange(44,66.5,0.5)[::-1]
    lon = np.arange(220,260.5,0.5)
    gefs = xr.open_dataset('data/gh-2017-12-22-2018-12-27-0.nc').isel(step=step)

    obs, gefs = prep_obs(xr.open_dataset('data/bm-2018-12h.nc'), gefs, step)
    

    gefs = prep_gefs(gefs, step)

    Nx = 3
    Ny = 10

    z, ind, bmus = train_som(gefs, obs)
    #plot_som(Nx, Ny, z, ind)
    wind_distributions(bmus)
    
