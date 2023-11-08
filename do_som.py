from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from som_class import SOM, BMUs, BMU_frequency
import xarray as xr
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import pickle
from cmcrameri import cm
import os


def prep_gefs(ds, obs_arr, step_int=0):
    bad_indices = np.argwhere(np.isnan(ds.gh.values))[:,0][0] # attempt at getting rid of nans TO DO OBS TOO!!!!
    if bad_indices:
        ds = ds.where(ds.time!=ds.time[bad_indices],drop=True)
        obs_arr = np.delete(obs_arr,bad_indices)

    ds_arr = ds.gh.to_numpy()
    ds_arr = np.reshape(ds_arr,(ds.time.shape[0],ds.latitude.shape[0]*ds.longitude.shape[0])) #(time,space)

    if ds_arr.shape[0] != obs_arr.shape[0]:
        print('GEFS and obs not the same shape! Exiting...')
        os.exit()


    return ds_arr, obs_arr



def prep_obs(ds, gefs, step_int):
    # make times of obs and gefs line up
 
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
    # I think I need to normalize over space, so each z pattern is conserved
    gefs_arr = (gefs_arr - np.mean(gefs_arr,axis=0)) / np.std(gefs_arr,axis=0)  # axis 1 is the space axis
    obs_arr = (obs_arr - np.mean(obs_arr)) / np.std(obs_arr)

    #pca_gefs(gefs_arr)

    learning_rate = 1e-2
    N_epochs = 60
    colours_list = 'default2'

    som = SOM(Nx, Ny, gefs_arr, N_epochs, linewidth=4, colours_list=colours_list)
    som.initialize_map(node_shape='hex')

    # train
    tic = time.perf_counter()
    som.train_map(learning_rate)
    toc = time.perf_counter()
    print(f'Finished training map in {toc - tic:0.2f} seconds. Saving...')

    # with open('trained-map.pkl', 'wb') as handle:
    #     pickle.dump(som, handle)

    return som


def plot_som(Nx, Ny, z, indices):
    proj=ccrs.PlateCarree()
    fig, axes = plt.subplots(nrows=Ny, ncols=Nx,sharex=True,sharey='row',figsize=(Nx*3,Ny*2),subplot_kw={'projection': proj, 'aspect':1.4},gridspec_kw = {'wspace':0.1, 'hspace':0.1})
    i = 0
    k = 3645
    for kk, ax in enumerate(axes.flatten()):
        var = z[indices[kk],i:k].reshape(45,81)
        ax.set_extent(([219,261,43.25,65.25]))
        
        
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_0_boundary_lines_land', scale='50m', facecolor='none', edgecolor='k',alpha=0.4,linewidth=0.6))
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_1_states_provinces_lines', scale='50m',facecolor='none', edgecolor='k',alpha=0.3,linewidth=0.6))
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', \
            scale='50m', edgecolor='k', facecolor='none', alpha=0.4,linewidth=0.6))
        
        ax.contourf(lon, lat, var, transform=ccrs.PlateCarree(),cmap=cm.acton)
        ax.scatter(360-120.4306,55.6986,c='k',transform=ccrs.PlateCarree(),s=6,marker='*')

        # Create gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.8, color='black', alpha=0.2,linestyle='--')
        # Manipulate gridlines number and spaces
        gl.ylocator = mticker.FixedLocator(np.arange(-90,90,10))
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10)) 
        if kk > (Ny*Nx) - Nx - 1:
            gl.bottom_labels = True
        if kk % Nx == 0:
            gl.left_labels = True
    plt.suptitle('z500 clusters')
    plt.show()

    #plt.plot(range(gefs_arr.shape[0]), bmus, 'bo--')
    return None


def wind_distributions(bmus):
    
    distributions = np.empty(N_nodes)
    
    fig, axes = plt.subplots(nrows=Ny, ncols=Nx)
    
    for i, ax in enumerate(axes.flatten()):
        distribution = obs[np.where(bmus==i+1)[0]]  # wind obs that belong to this node -> this is our distribution
        ax.hist(distribution, bins='auto')
        ax.set_title('Avg wind speed ='+str(round(np.mean(distribution),2))+'m/s')

    plt.tight_layout()
    plt.show()

    return distributions



if __name__ ==  "__main__":
    step = 3
    lat = np.arange(44,66.5,0.5)[::-1]
    lon = np.arange(220,260.5,0.5)
    #gefs = xr.open_dataset('/Users/jpsotka/Nextcloud/geo-height-data/gh-2012-11-20-2017-12-25-0.nc')#.isel(step=step)
    gefs = xr.open_dataset('data/geo-height-2012-2017-12h-0.nc').isel(step=step)

    obs, gefs = prep_obs(xr.open_dataset('data/obs-all-12h.nc'), gefs, step)
    

    gefs, obs = prep_gefs(gefs, obs, step)

    Nx = 2
    Ny = 6
    N_nodes = Nx * Ny

    #som = train_som(gefs, obs)
    with open('trained-map.pkl','rb') as handle:
        som = pickle.load(handle)
    z = som.z  # pattern of each node
    indices = np.arange(N_nodes).reshape(Nx,Ny).T.flatten()
    bmus = BMUs(som)  # the nodes that each forecast belongs to -> use this to look at wind
    freq = BMU_frequency(som)  # frequency of each node
    QE = som.QE()  # quantization error
    TE = som.TE()  # topographic error
    plot_som(Nx, Ny, z, indices)
    #wind_distributions(bmus)
    
