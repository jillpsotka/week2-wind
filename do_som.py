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
import matplotlib.cm as comap
import os
from resampling import era5_prep, resample_mean, low_pass_filter, dates_obs_gefs
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8


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


def visualize_normalization(gefs_arr):
    gefs_mean = np.mean(era)  # this is the mean of each time step
    gefs_std = np.std(era)
    gefs_arr1 = (gefs_arr - gefs_mean) / gefs_std  # normalizing each time frame
    gefs_arr2 = (gefs_arr1 * gefs_std) + gefs_mean  # reconstructed
    # normalization from 510 did the whole array not just one axis - anomaly vibes
    fig, axes = plt.subplots(nrows=1, ncols=3,sharex=True,sharey='row',figsize=(6,4))

    for kk, ax in enumerate(axes.flatten()):
        if kk == 0:
            var = gefs_arr[kk,:].reshape(45,81)
        elif kk == 1:
            var = gefs_arr2[0,:].reshape(45,81)
        else:
            var = gefs_arr1[0,:].reshape(45,81)
    
        cs = ax.contourf(lon, lat, var,cmap=cm.acton)
        ax.scatter(360-120.4306,55.6986,c='k',s=6,marker='*')
        fig.colorbar(cs)

    plt.suptitle('z500 clusters')
    plt.show()



def train_som(gefs_arr, obs_arr):

    #pca_gefs(gefs_arr)

    learning_rate = 1e-4
    N_epochs = 50
    colours_list = 'default2'

    som = SOM(Nx, Ny, gefs_arr, N_epochs, linewidth=4, colours_list=colours_list)
    som.initialize_map(node_shape='hex')

    # train
    tic = time.perf_counter()
    som.train_map(learning_rate)
    toc = time.perf_counter()
    print(f'Finished training map in {(toc - tic)/60:0.2f} minutes. Saving...')
    som.z_raw = som.z*gefs_std + gefs_mean

    with open('trained-map-'+title, 'wb') as handle:
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


def test_shapes(gefs_arr):
    testing_x = np.arange(3,12)
    testing_y = np.arange(2,7)
    x_labels=[]

    learning_rate = 1e-3
    N_epochs = 200
    colours_list = 'pink_blue_red_purple'

    QE = []
    TE = []
    for ii in range(testing_x.shape[0]):
        Nx = testing_x[ii]
        for jj in range(testing_y.shape[0]):
            x_labels.append(str(testing_x[ii])+'-'+str(testing_y[jj]))
            Ny = testing_y[jj]
            som = SOM(Nx, Ny, gefs_arr, N_epochs, linewidth = 4, colours_list = colours_list)
            som.initialize_map(node_shape = 'hex')
            som.train_map(learning_rate)
            z = som.z #this is the pattern of each BMU
            QE.append(som.QE()) #quantization error of map
            TE.append(som.TE()) #topographic error of map
        print(ii,'out of ',testing_x.shape[0])
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title('QE and TE')
    ax1.plot(np.arange(testing_x.shape[0]*testing_y.shape[0]),QE,label='QE')
    ax2.plot(np.arange(testing_x.shape[0]*testing_y.shape[0]),TE,label='TE',c='orange')
    ax1.set_xticks(np.arange(testing_x.shape[0]*testing_y.shape[0]))
    ax1.set_xticklabels(x_labels)
    ax1.legend(loc=0)
    ax2.legend()
    plt.show()
    #plt.savefig('QEVTE.png')



def plot_som(Nx, Ny, z, indices):
    proj=ccrs.PlateCarree()
    vmin = np.min(z)
    vmax = np.max(z)  # colorbar range
    fig, axes = plt.subplots(nrows=Ny, ncols=Nx,sharex=True,sharey='row',layout='constrained',figsize=(Nx*4,Ny*4),subplot_kw={'projection': proj, 'aspect':1.5},gridspec_kw = {'wspace':0.03, 'hspace':0.1})
    im = comap.ScalarMappable(norm=Normalize(vmin,vmax),cmap=cm.acton)
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
        gl.xlabel_style = {'size': 8,'rotation':25}
        gl.ylabel_style = {'size': 8} 
        if kk > (Ny*Nx) - Nx - 1:
            gl.bottom_labels = True
        if kk % Nx == 0:
            gl.left_labels = True

    #cbar_ax = fig.add_axes([0.05, 0.07, 0.45*Nx, 0.03])
    cbar = fig.colorbar(im,ax=axes,fraction=0.046, pad=0.04,orientation='horizontal')
    if anomaly:
        cbar.set_label('z500 anomaly (m)')
    else:
        cbar.set_label('z500 (m)')
    plt.suptitle('z500 clusters')
    plt.show()

    return None


def wind_distributions(bmus):
    
    distributions = []
    
    fig, axes = plt.subplots(nrows=Ny, ncols=Nx, gridspec_kw = {'wspace':0.5, 'hspace':0.5})
    vmin = np.min(obs)
    vmax = np.max(obs)
    
    for i, ax in enumerate(axes.flatten()):
        distribution = obs[np.where(bmus==i)[0]]  # wind obs that belong to this node -> this is our distribution
        ax.hist(distribution, range=(vmin,vmax),bins='auto')
        ax.set_title('Mean ='+str(round(np.mean(distribution),2))+'m/s, Median ='+str(round(np.median(distribution),2))+'m/s')
        distributions.append(distribution)

    #plt.tight_layout()
    plt.show()

    with open('distributions-'+title,'wb') as f:
        pickle.dump(distributions,f)

    return distributions




if __name__ ==  "__main__":

    # setup
    res = 48  # time resolution in hours
    Nx = 2
    Ny = 2
    N_nodes = Nx * Ny
    title = '48h-2x2-anomalies-all.pkl'
    period=slice("2009-10-01","2020-10-01")
    lat = np.arange(44,66.5,0.5)[::-1]
    lon = np.arange(220,260.5,0.5)
    train = False
    anomaly = True  # get rid of seasonal anomaly using 30-day rolling avg

    # data
    print('Prepping data...')

    obs = xr.open_dataset('data/bm_cleaned_all.nc').sel(index=period)
    obs = low_pass_filter(obs,'obs',res)

    era = xr.open_dataset('era-2009-2022.nc').sel(time=period)
    era = low_pass_filter(era,'era',res)

    if anomaly:
        #smoothed = era.rolling(time=int(5*(24/res)),center=True).mean()
        #smoothed = smoothed.rolling(time=int(31*(24/res)),center=True).mean()
        clim = era.groupby("time.dayofyear").mean(dim=["time"])
        era = era.groupby("time.dayofyear") - clim

    obs, era = dates_obs_gefs(obs, era)
    obs, era = prep_data(era, obs)

    if train:

        # normalize data (this is actually the z score)
        gefs_mean = np.mean(era) 
        gefs_std = np.std(era)
        era = (era - gefs_mean) / gefs_std  # normalizing each time frame

        #test_shapes(era)

        print('Training map...')
        som = train_som(era, obs)
    else:
        with open('trained-map-'+title,'rb') as handle:
            som = pickle.load(handle)

    #monitor_training(som)
    
    indices = np.arange(N_nodes).reshape(Nx,Ny).T.flatten()
    bmus = BMUs(som)  # the nodes that each forecast belongs to -> use this to look at wind
    wind_distributions(bmus)
    freq = BMU_frequency(som)  # frequency of each node
    QE = som.QE()  # quantization error
    TE = som.TE()  # topographic error

    plot_som(Nx, Ny, som.z_raw, indices)
    #plt.plot(range(era.shape[0]), bmus, 'bo--')
    #plt.show()
    
