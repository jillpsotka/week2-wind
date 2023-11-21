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
    bad_indices = np.argwhere(np.isnan(ds.gh.values))[:,0][0] 
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
    # I checked: it is correct that gefs is in form (time, space)
    # this makes the eigvecs the same shape as the grid, and the PCs have length time
    pca = PCA(n_components=10)
    PCs = pca.fit_transform(gefs)
    frac_var = pca.explained_variance_ratio_
    var = pca.explained_variance_
    std = var ** 0.5
    eigvecs = pca.components_


    return None



def train_som(gefs_arr, obs_arr):
    # normalize data (this is actually the z score, could try other methods of standardization)
    # TO DO: leaving this out for now, since normalizing feels almost like taking anomalies, which I don't want to do
    # -> in the future, I'll try normalizing over both space and time and see how it affects performance
    # I think I need to normalize over space, so each z pattern is conserved
    # normalization across time: anomalies. normalization across space: chill?
    gefs_mean = np.mean(gefs)
    gefs_std = np.std(gefs)
    #gefs_arr = (gefs_arr - np.mean(gefs_arr,axis=0)) / np.std(gefs_arr,axis=0)  # axis 1 is the space axis
    # TO DO: Look at normalization from 510 lab 8 

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

    with open('trained-map.pkl', 'wb') as handle:
        pickle.dump(som, handle)

    return som


def test_shapes(gefs_arr):
    STOP = 0
    testing_x = np.arange(1,2)
    testing_y = np.arange(2,4)
    learning_rate = 1e-2
    N_epochs = 100
    colours_list = 'pink_blue_red_purple'
    colours_list = 'pinks'
    colours_list = 'default2'
    QE = []
    TE = []
    for ii in range(testing_x.shape[0]):
        Nx = testing_x[ii]
        for jj in range(testing_y.shape[0]):
            Ny = testing_y[jj]
            som = SOM(Nx, Ny, gefs_arr, N_epochs, linewidth = 4, colours_list = colours_list)
            som.initialize_map(node_shape = 'hex')
            som.train_map(learning_rate)
            z = som.z #this is the pattern of each BMU
            QE.append(som.QE()) #quantization error of map
            TE.append(som.TE()) #topographic error of map
    
    plt.figure()
    plt.title('QE and TE')
    plt.plot(np.arange(testing_x.shape[0]*testing_y.shape[0]),QE,label='QE')
    plt.plot(np.arange(testing_x.shape[0]*testing_y.shape[0]),TE,label='TE')
    plt.legend()
    plt.show()
    #plt.savefig('QEVTE.png')

    stop = 0


def plot_som(Nx, Ny, z, indices):
    proj=ccrs.PlateCarree()
    vmin = np.min(z)
    vmax = np.max(z)  # colorbar range
    fig, axes = plt.subplots(nrows=Ny, ncols=Nx,sharex=True,sharey='row',figsize=(Nx*3,Ny*2),subplot_kw={'projection': proj, 'aspect':1.4},gridspec_kw = {'wspace':0.3, 'hspace':0.05})

    for kk, ax in enumerate(axes.flatten()):
        var = z[indices[kk],:].reshape(45,81)
        ax.set_extent(([219,261,43.25,65.25]))
        ax.set_title('Node '+str(indices[kk]))      
        
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_0_boundary_lines_land', scale='50m', facecolor='none', edgecolor='k',alpha=0.4,linewidth=0.6))
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_1_states_provinces_lines', scale='50m',facecolor='none', edgecolor='k',alpha=0.3,linewidth=0.6))
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', \
            scale='50m', edgecolor='k', facecolor='none', alpha=0.4,linewidth=0.6))
        
        cs = ax.contourf(lon, lat, var, vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),cmap=cm.acton)
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

    cbar_ax = fig.add_axes([0.1, 0.05, 0.6, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax,orientation='horizontal')
    plt.suptitle('z500 clusters')
    plt.show()

    
    return None


def wind_distributions(bmus):
    
    distributions = np.empty(N_nodes)
    
    fig, axes = plt.subplots(nrows=Ny, ncols=Nx, gridspec_kw = {'wspace':0.5, 'hspace':0.5})
    vmin = np.min(obs)
    vmax = np.max(obs)
    
    for i, ax in enumerate(axes.flatten()):
        distribution = obs[np.where(bmus==i)[0]]  # wind obs that belong to this node -> this is our distribution
        ax.hist(distribution, range=(vmin,vmax),bins='auto')
        ax.set_title('Avg wind speed ='+str(round(np.mean(distribution),2))+'m/s')

    #plt.tight_layout()
    plt.show()

    return distributions



if __name__ ==  "__main__":
    step = 3
    lat = np.arange(44,66.5,0.5)[::-1]
    lon = np.arange(220,260.5,0.5)
    #gefs = xr.open_dataset('/Users/jpsotka/Nextcloud/geo-height-data/gh-2012-11-20-2017-12-25-0.nc')#.isel(step=step)
    #gefs = xr.open_dataset('data/geo-height-2012-2017-12h-0.nc').isel(step=step)
    gefs = xr.open_dataset('data/gh-reanalysis-2014-01.nc')

    obs, gefs = prep_obs(xr.open_dataset('data/obs-all-12h.nc'), gefs, step)
    
    gefs, obs = prep_gefs(gefs, obs, step)

    Nx = 6
    Ny = 2
    N_nodes = Nx * Ny
    test_shapes(gefs)
    train = False
    if train:
        som = train_som(gefs, obs)
    else:
        with open('trained-map.pkl','rb') as handle:
            som = pickle.load(handle)

    z = som.z  # pattern of each node
    z_epochs = som.z_epochs  # pattern of each node through training
    indices = np.arange(N_nodes).reshape(Nx,Ny).T.flatten()
    bmus = BMUs(som)  # the nodes that each forecast belongs to -> use this to look at wind
    freq = BMU_frequency(som)  # frequency of each node
    QE = som.QE()  # quantization error
    TE = som.TE()  # topographic error
    plot_som(Nx, Ny, z, indices)
    #plt.plot(range(gefs.shape[0]), bmus, 'bo--')
    #plt.show()
    wind_distributions(bmus)
    
