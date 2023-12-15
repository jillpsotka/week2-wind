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
from resampling import gefs_reanalysis, cleaned_obs, dates_obs_gefs


def prep_data(ds, obs):

    ds_arr = ds.gh.to_numpy()
    obs_arr = obs.Wind.to_numpy()
    ds_arr = np.reshape(ds_arr,(ds.time.shape[0],ds.latitude.shape[0]*ds.longitude.shape[0])) #(time,space)

    if ds_arr.shape[0] != obs_arr.shape[0]:
        print('GEFS and obs not the same shape! Exiting...')
        os.exit()


    return obs_arr, ds_arr




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


def visualize_normalization(gefs_arr):
    gefs_mean = np.mean(gefs)  # this is the mean of each time step
    gefs_std = np.std(gefs)
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
    # normalize data (this is actually the z score, could try other methods of standardization)
    # TO DO: much to think about
    gefs_mean = np.mean(gefs) 
    gefs_std = np.std(gefs)
    gefs_arr = (gefs_arr - gefs_mean) / gefs_std  # normalizing each time frame
    # normalization from 510 did the whole array

    #pca_gefs(gefs_arr)

    learning_rate = 1e-3
    N_epochs = 200
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


def monitor_training(som):
    z_epochs = som.z_epochs
    diff = []
    for i in range(1,z_epochs.shape[2]):
        update = np.sum(z_epochs[:,:,i] - z_epochs[:,:,i-1])
        diff.append(update)

    plt.plot(diff)
    plt.show()
    
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
    cbar.set_label('z500')
    plt.suptitle('z500 clusters')
    plt.show()

    
    return None



if __name__ ==  "__main__":

    lat = np.arange(44,66.5,0.5)[::-1]
    lon = np.arange(220,260.5,0.5)
    #gefs = xr.open_dataset('/Users/jpsotka/Nextcloud/geo-height-data/gh-2012-11-20-2017-12-25-0.nc')#.isel(step=step)
    #gefs1 = xr.open_dataset('data/geo-height-2012-2017-12h-0.nc').isel(step=step)
    

    obs = cleaned_obs(res=6)
    gefs = xr.open_dataset('gh-reanalysis-all-2012-2019.nc').sel(isobaricInhPa=500)
    gefs = gefs_reanalysis(gefs)
    obs, gefs = dates_obs_gefs(obs, gefs)
    obs, gefs = prep_data(gefs, obs)

    gefs_mean = np.mean(gefs) 
    gefs_std = np.std(gefs)
    gefs = (gefs - gefs_mean) / gefs_std  # normalizing each time frame

    Nx = 6
    Ny = 3
    N_nodes = Nx * Ny
    test_shapes(gefs)
    train = False
    if train:
        som = train_som(gefs, obs)
    else:
        with open('trained-map.pkl','rb') as handle:
            som = pickle.load(handle)

    #monitor_training(som)

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
    
