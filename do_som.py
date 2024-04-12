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
from scipy import signal
import scipy.stats as stats


mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12


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
    som.z_raw = som.z*era_std + era_mean

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



def plot_som(Nx, Ny, z, indices):
    proj=ccrs.PlateCarree()
    vmin = np.min(z)
    vmax = np.max(z)  # colorbar range
    fig, axes = plt.subplots(nrows=Ny, ncols=Nx,sharex=True,sharey='row',layout='constrained',figsize=(Nx*2,Ny*2),subplot_kw={'projection': proj, 'aspect':1.5},gridspec_kw = {'wspace':0.005, 'hspace':0.05})
    im = comap.ScalarMappable(norm=Normalize(vmin,vmax),cmap=cm.lipari)
    for kk, ax in enumerate(axes.flatten()):
        var = z[indices[kk],:].reshape(lat.shape[0],lon.shape[0])
        ax.set_extent(([219,261,43.25,65.25]))
        ax.set_title('Node '+str(indices[kk]))      
        
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_0_boundary_lines_land', scale='50m', facecolor='none', edgecolor='k',alpha=0.4,linewidth=0.6))
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_1_states_provinces_lines', scale='50m',facecolor='none', edgecolor='k',alpha=0.3,linewidth=0.6))
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', \
            scale='50m', edgecolor='k', facecolor='none', alpha=0.4,linewidth=0.6))
        
        cs = ax.contourf(lon, lat, var, vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),cmap=cm.lipari)
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
    cbar = fig.colorbar(im,ax=axes,fraction=0.06, pad=0.04,orientation='horizontal')
    if anomaly:
        cbar.set_label('50kPa anomaly (m)')
    else:
        cbar.set_label('50 kPa (m)')
    plt.suptitle('50 kPa clusters')
    plt.savefig('plots/current-som.png',dpi=200)
    #plt.show()


    return None


def wind_distributions(bmus):
    
    distributions = []
    
    fig, axes = plt.subplots(nrows=Ny, ncols=Nx, figsize=(Nx*2,Ny*2),gridspec_kw = {'wspace':0.5, 'hspace':0.5})
    vmin = np.min(obs)
    vmax = np.max(obs)
    
    for i, ax in enumerate(axes.flatten()):
        distribution = obs[np.where(bmus==i)[0]]  # wind obs that belong to this node -> this is our distribution
        ax.hist(distribution, range=(vmin,vmax),bins='auto',color='black')
        ax.set_title('Me='+str(round(np.mean(distribution),1))+'(m/s)')
        ind = int((np.mean(distribution)-3)*(255)/(10-3))
        col_ar = cm.lipari.colors[ind]
        col_tuple = tuple(col_ar)
        ax.patch.set_facecolor(col_tuple)
        ax.patch.set_alpha(0.9)
        distributions.append(distribution)

    #plt.tight_layout()
    plt.savefig('plots/current-som-dist.png',dpi=200)
    #plt.show()

    with open('distributions-'+title,'wb') as f:
        pickle.dump(distributions,f)

    return distributions




if __name__ ==  "__main__":

    # setup
    res = 24  # time resolution in hours
    Nx = 6
    Ny = 2
    N_nodes = Nx * Ny
    title = '24h-6x2-anomalies-500.pkl'
    period=slice("2009-10-01","2020-09-30")
    lat_offset = 11
    lon_offset = 20
    lat = np.arange(55-lat_offset,55+lat_offset+0.5,0.5)[::-1]
    lon = np.arange(240-lon_offset,240+lon_offset+0.5,0.5)
    train = False
    anomaly = True  # get rid of seasonal anomaly using 30-day rolling avg
    use_wind = False  # concat obs into training data to guide map
    number = 0 # weiht of wind

    # data
    print('Prepping data...')

    obs = xr.open_dataset('~/Nextcloud/thesis/bm_cleaned_all.nc').sel(index=period)
    obs = resample_mean(obs,'obs',res)

    era = xr.open_dataset('era-2009-2022.nc').sel(latitude=lat,longitude=lon-360,time=period,level=700)
    era = resample_mean(era,'era',res)

    if anomaly:
        clim = era.groupby("time.dayofyear").mean(dim=["time"])
        clim_concat = xr.concat([clim.isel(dayofyear=slice(330,366)),clim,clim.isel(dayofyear=slice(0,36))],dim='dayofyear')
        cutoff=0.03
        b, a = signal.butter(5, cutoff, btype='lowpass')
        dUfilt = signal.filtfilt(b, a, clim_concat.gh.values,axis=0)
        dUfilt = dUfilt[36:-36,:,:]
        clim.gh.values = dUfilt
        era = era.groupby("time.dayofyear") - clim

    obs, era = dates_obs_gefs(obs, era)
    obs, era = prep_data(era, obs)

    # normalize data (this is actually the z score)
    era_mean = np.mean(era) 
    era_std = np.std(era)
    era = (era - era_mean) / era_std  # normalizing each time frame

    if use_wind:  # add wind obs to training to try to optimize
        obs_repeat = np.repeat(obs[:,np.newaxis],number,axis=1)  # (time, repeats)
        era = np.concatenate([era,obs_repeat],axis=1) # adding repeated obs onto end of lat/lon data


    if train:
        #test_shapes(era)

        print('Training map...')
        som = train_som(era, obs)
    else:
        with open('trained-map-'+title,'rb') as handle:
            som = pickle.load(handle)

    #monitor_training(som)
    
    indices = np.arange(N_nodes).reshape(Nx,Ny).T.flatten()
    bmus = BMUs(som)  # the nodes that each forecast belongs to -> use this to look at wind
    distributions = wind_distributions(bmus)
    freq = BMU_frequency(som)  # frequency of each node
    QE = som.QE()  # quantization error
    TE = som.TE()  # topographic error

    if use_wind:
        som.z_raw = som.z_raw[:,:-number]
    plot_som(Nx, Ny, som.z_raw, indices)

    # some stats on how good the clustering is

    WSS_nodes = np.empty(N_nodes)
    dist_means = np.empty(N_nodes)
    BSS_nodes = np.empty(N_nodes)

    for i in range(len(distributions)):  # for each node
        mean = np.mean(distributions[i])
        WSS_nodes[i] = np.sum(np.square(distributions[i] - mean))
        dist_means[i] = mean

    WSS = np.sum(WSS_nodes)
    TSS = np.sum(np.square(obs - np.mean(dist_means)))
    EV = 1 - WSS/TSS  # explained variance

    sig_count = 0
    for i in range(len(distributions)):
        BSS_nodes[i] = np.square(dist_means[i] - np.mean(dist_means))
        # K-S test
        other = obs[np.where(bmus!=i)[0]]  # the rest of the obs that are not in this node
        n = distributions[i].shape[0]
        m = other.shape[0]
        crit = 1.36*np.sqrt((n+m)/(n*m))  # 1.36 is for 95% confidence, 1.07 for 80 1.22 for 90 1.52 for 98 1.63 for 99
        ks = stats.ks_2samp(distributions[i], other)  # rejection means the distirbutions are different
        if ks.statistic > crit and ks.pvalue < 0.05:  # rejection of null
            sig_count +=1

    ks_sig_frac = sig_count / N_nodes  # perentage of nodes that have significantly different distribution
    n = obs.shape[0]
    PF = (np.sum(BSS_nodes)/(N_nodes-1)) / (WSS/(n-N_nodes))  # pseudo-F statistic
    print(' EV:',EV,' PF:',PF,' K-S fraction siginifcant:',ks_sig_frac,' TE:',TE)
    #plt.plot(range(era.shape[0]), bmus, 'bo--')
    #plt.show()
    
