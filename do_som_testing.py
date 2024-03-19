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
from resampling import resample_mean, low_pass_filter, dates_obs_gefs
from matplotlib.colors import Normalize
import matplotlib as mpl
from sklearn.cluster import KMeans
from scipy import signal
import scipy.stats as stats



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
    som.train_map(learning_rate)

    som.z_raw = som.z*gefs_std + gefs_mean

    # with open('trained-map-'+title, 'wb') as handle:
    #     pickle.dump(som, handle)

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
    N_epochs = 50
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
    #vmin = np.min(obs)
    #vmax = np.max(obs)
    
    for i, ax in enumerate(axes.flatten()):
        distribution = obs[np.where(bmus==i)[0]]  # wind obs that belong to this node -> this is our distribution
        #ax.hist(distribution, range=(vmin,vmax),bins='auto')
        #ax.set_title('Mean ='+str(round(np.mean(distribution),2))+'m/s, Median ='+str(round(np.median(distribution),2))+'m/s')
        distributions.append(distribution)
    plt.close(fig)

    #plt.tight_layout()
    #plt.show()

    # with open('distributions-'+title,'wb') as f:
    #     pickle.dump(distributions,f)

    return distributions




if __name__ ==  "__main__":
    # setup
    sizes = [(5,3),(5,4),(5,5),(6,2),(6,3),(6,4),(6,5),(6,6),(7,3),(7,4),(7,5),(7,6),(7,7),(8,3),(8,4),(8,5),(8,6),(8,7),(8,8),(9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,9)]  # map size
    lat_dif = [9,11]  # domain size (degrees on each side of the center)
    lon_dif = [18,20]
    res = 24  # time resolution in hours
    k_m = False

    EV_max = 0
    PF_max = 0
    EV_list = []
    PF_list = []
    TE_list = []
    KS_list = []
    anomaly = True  # get rid of seasonal anomaly using 30-day rolling avg
    period=slice("2009-10-01","2020-10-01")
    obs_full = xr.open_dataset('~/Nextcloud/thesis/bm_cleaned_all.nc').sel(index=period)
    obs_full = low_pass_filter(obs_full,'obs',res)
    era_full = xr.open_dataset('era-2009-2022-500.nc').sel(time=period)

    title = '24h-anomalies-som'

    with open('stats-'+title+'.txt', 'w') as file:
        file.write('Nx,Ny,lat,lon,TE,EV,PF,KS frac')

    for dom in range(len(lat_dif)):
        tic = time.perf_counter()

        lat_offset = lat_dif[dom]
        lon_offset = lon_dif[dom]
        lat = np.arange(55-lat_offset,55+lat_offset+0.5,0.5)[::-1]
        lon = np.arange(240-lon_offset,240+lon_offset+0.5,0.5)

        era = era_full.sel(latitude=lat,longitude=lon-360)
        era = low_pass_filter(era,'era',res)

        if anomaly:
            clim = era.groupby("time.dayofyear").mean(dim=["time"])
            clim_concat = xr.concat([clim.isel(dayofyear=slice(330,366)),clim,clim.isel(dayofyear=slice(0,36))],dim='dayofyear')
            cutoff=0.03
            b, a = signal.butter(5, cutoff, btype='lowpass')
            dUfilt = signal.filtfilt(b, a, clim_concat.gh.values,axis=0)
            dUfilt = dUfilt[36:-36,:,:]
            clim.gh.values = dUfilt

            era = era.groupby("time.dayofyear") - clim

        obs, era = dates_obs_gefs(obs_full, era)
        obs, era = prep_data(era, obs)

        # normalize data (this is actually the z score)
        gefs_mean = np.mean(era) 
        gefs_std = np.std(era)
        era = (era - gefs_mean) / gefs_std  # normalizing each time frame
        obs_mean = np.mean(obs) 
        obs_std = np.std(obs)
        obs = (obs - obs_mean) / obs_std  # normalizing each time frame

        for (Nx, Ny) in sizes:
            N_nodes = Nx * Ny

            if k_m:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(era)

                #you can see the labels with:
                print(kmeans.labels_)

                # the output will be something like:
                #array([0, 0, 0, 1, 1, 1], dtype=int32)
                # the values (0,1) tell you to what cluster does every of your data points correspond to

                #or see were the centres of your clusters are
                kmeans.cluster_centers_

            som = train_som(era, obs)

            #monitor_training(som)
            
            indices = np.arange(N_nodes).reshape(Nx,Ny).T.flatten()
            bmus = BMUs(som)  # the nodes that each forecast belongs to -> use this to look at wind
            obs = (obs*obs_std) + obs_mean
            distributions = wind_distributions(bmus)
            freq = BMU_frequency(som)  # frequency of each node
            QE = som.QE()  # quantization error
            TE = som.TE()  # topographic error

            #plot_som(Nx, Ny, som.z_raw[:,:-number], indices)

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

            if EV > EV_max and PF > PF_max:
                print('New maximums found. Num nodes ',N_nodes, ', and lat offset', lat_offset)
                EV_max = EV
                PF_max = PF


            with open('stats-'+title+'.txt','a') as file:
                file.write('\n'+str(Nx)+','+str(Ny)+','+str(lat_offset)+','+str(lon_offset)+','+str(TE)+','+str(EV)+','+str(PF)+','+str(ks_sig_frac))
            EV_list.append(EV)
            PF_list.append(PF)
            TE_list.append(TE)
            KS_list.append(ks_sig_frac)
        
        toc = time.perf_counter()
        print('Done that domain size',dom+1,f'/3 in {(toc - tic)/60:0.2f} minutes.')
    fig, ax = plt.subplots(1)
    ax.plot(EV_list,label='EV')
    ax.plot(PF_list,label='PF')
    ax.plot(TE_list,label='TE')
    plt.legend()
    ax2 = ax.twinx()
    ax2.plot(KS_list,color='black')
    plt.savefig('map-error-testing-'+title+'.png')

        
