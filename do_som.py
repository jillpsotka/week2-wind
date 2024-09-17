from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from som_class import SOM, BMUs, BMU_frequency
import xarray as xr
import time
import pickle
import os
from resampling import resample_mean, dates_obs_gefs
import matplotlib as mpl
from scipy import signal
from datetime import datetime
import glob
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def prep_data(ds, obs):

    ds_arr = ds.gh.to_numpy()
    obs_arr = obs.Wind.to_numpy()
    ds_arr = np.reshape(ds_arr,(ds.time.shape[0],ds.latitude.shape[0]*ds.longitude.shape[0])) #(time,space)

    if ds_arr.shape[0] != obs_arr.shape[0]:
        print('GEFS and obs not the same shape! Exiting...')
        os.exit()


    return obs_arr, ds_arr


def pca_gefs(era):
    # do PCA to see the ratio of nodes in the ideal map
    # I checked: it is correct that era is in form (time, space)
    # this makes the eigvecs the same shape as the grid, and the PCs have length time
    pca = PCA(n_components=10)
    PCs = pca.fit_transform(era)
    frac_var = pca.explained_variance_ratio_
    var = pca.explained_variance_
    std = var ** 0.5
    eigvecs = pca.components_


    return None


def train_som(gefs_arr):

    #pca_gefs(gefs_arr)

    learning_rate = 1e-4
    N_epochs = 50
    colours_list = 'default2'

    som = SOM(Nx, Ny, gefs_arr, N_epochs, linewidth=4, colours_list=colours_list)
    som.initialize_map(node_shape='hex')

    # train
    som.train_map(learning_rate)

    som.z_raw = som.z*era_std + era_mean

    with open('trained-map-'+title+'-final.pkl', 'wb') as handle:
        pickle.dump(som, handle)

    return som


def monitor_training(som):
    z_epochs = som.z_epochs
    diff = []
    for i in range(1,z_epochs.shape[2]):
        update = np.sum(np.square(z_epochs[:,:,i] - z_epochs[:,:,i-1]))
        diff.append(update)
    plt.plot(diff)
    plt.show()
    print(np.min(np.array(diff)))
    
    return None


def wind_distributions(bmus):
    
    distributions = []
    axes = np.empty((Nx,Ny))
    
    for i, ax in enumerate(axes.flatten()):
        distribution = obs[np.where(bmus==i)[0]]  # wind obs that belong to this node -> this is our distribution
        distributions.append(distribution)

    with open('distributions-'+title+'-final.pkl','wb') as f:
        pickle.dump(distributions,f)

    return distributions


if __name__ ==  "__main__":
    # setup
    print('starting',datetime.now())
    Nx = 15
    Ny = 2

    seas = 'DJF'
    levels = [1000]
    lat_dif = [9]  # domain size (degrees on each side of the center)
    lon_dif = [16]

    title = seas + '-' + str(levels[0])

    res = 6  # time resolution of map in hours, always 6
    anomaly = True  # get rid of seasonal anomaly
    train_period=slice("2009-10-01","2020-10-01")

    print('Loading data...')

    obs_full = xr.open_dataset('~/Nextcloud/thesis/bm_cleaned_all.nc')
    obs_full = resample_mean(obs_full,'obs',res)
    obs_full = obs_full.sel(index=obs_full.index.dt.season==seas)
    obs_train = obs_full.sel(index=train_period)

    obs_full = None

    for level in levels:
            
        for dom in range(len(lat_dif)):  # for each domain size
            lat_offset = lat_dif[dom]
            lon_offset = lon_dif[dom]
            lat = np.arange(55-lat_offset,55+lat_offset+0.5,0.5)[::-1]
            lon = np.arange(240-lon_offset,240+lon_offset+0.5,0.5)

            if level == 850:
                era = xr.open_dataset('era-850-2009-2022.nc').sel(latitude=lat,longitude=lon-360)
            else:
                era = xr.open_dataset('era-2009-2022-a.nc').sel(latitude=lat,longitude=lon-360,level=level)

            era = era.sel(time=train_period)
            tic = time.perf_counter()

            if anomaly:
                print('Processing data...')
                # taking out all (spatial and temporal) anomaly
                clim = era.groupby("time.dayofyear").mean(dim=["time"])  # clim for each doy for each pixel
                clim_concat = xr.concat([clim.isel(dayofyear=slice(330,366)),clim,clim.isel(dayofyear=slice(0,36))],dim='dayofyear')
                cutoff=0.03
                b, a = signal.butter(5, cutoff, btype='lowpass')
                dUfilt = signal.filtfilt(b, a, clim_concat.gh.values,axis=0)
                dUfilt = dUfilt[36:-36,:,:]
                clim.gh.values = dUfilt

                era = era.groupby("time.dayofyear") - clim
            era = resample_mean(era,'era',res)

            obs, era = dates_obs_gefs(obs_train, era)
            obs, era = prep_data(era, obs)
            #a, erav = dates_obs_gefs(obsv, erav)
            #obsv = obsv.Wind.to_numpy()

            # normalize data (this is actually the z score)
            era_mean = np.mean(era)
            era_std = np.std(era)
            era = (era - era_mean) / era_std  # normalizing each time frame

            N_nodes = Nx * Ny

            print('\n training map...',Nx,'x',Ny,flush=True)
            som = train_som(era)
            
            indices = np.arange(N_nodes).reshape(Nx,Ny).T.flatten()
            bmus = BMUs(som)  # the nodes that each gh best matches
            distributions = wind_distributions(bmus)
            freq = BMU_frequency(som)  # frequency of each node

        print('Done',datetime.now())

        