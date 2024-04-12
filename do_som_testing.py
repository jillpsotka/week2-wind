from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from som_class import SOM, BMUs, BMU_frequency
import xarray as xr
import time
import properscoring as ps
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
from sklearn.metrics import brier_score_loss
import scipy.stats as stats
from datetime import datetime



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
    print('starting',datetime.now())
    x_to_try = np.arange(2,21)
    sizes = []
    for x in x_to_try:
        for y in range(1,x+1):
            if x*y < 41:
                sizes.append((x,y))

    lat_dif = [7,9,11]  # domain size (degrees on each side of the center)
    lon_dif = [12,16,20]
    res = 24  # time resolution in hours
    k_m = False
    t_step = np.array(int(7*24*1e9*60*60),dtype='timedelta64[ns]')

    anomaly = True  # get rid of seasonal anomaly using 30-day rolling avg
    train_period=slice("2009-10-01","2020-09-23")
    val_period = slice("2020-10-01","2022-03-31")
    val_period_gefs = slice("2020-09-24","2022-03-24")
    print('assembling data...')
    obs_full = xr.open_dataset('~/Nextcloud/thesis/bm_cleaned_all.nc')
    obs_full = resample_mean(obs_full,'obs',res)
    obs_train = obs_full.sel(index=train_period)
    obs_val = obs_full.sel(index=val_period)
    era_full = xr.open_dataset('era-2009-2022.nc').sel(level=500)
    era_train = era_full.sel(time=train_period)
    era_val = era_full.sel(time=val_period)
    gefs_val_full = xr.open_dataset('data/gefs-z-2020-09-24-2023-12-31-0-500.nc').sel(time=val_period_gefs)
    gefs_val_full = gefs_val_full.shift(time=7)  # now time is really valid time of 7-day forecast -> quick fix, not generalizable >:(
    gefs_val_full['longitude'] = gefs_val_full['longitude'] - 360

    title = '24h-anomalies-som-500-test'

    with open('stats-'+title+'.txt', 'w') as file:
        file.write('Nx,Ny,Nnodes,lat,lon,TE,QE,EV,PF,WSS,KS-frac,range,std,R,rmse,mae,bias,'+
                   'D,bss-25,bss-50,bss-75,bss-90,bss-95,crpss-nodes,mae-nodes,rmse-nodes,bias-nodes,'+
                   'R-gefs,rmse-gefs,mae-gefs,bias-gefs,D-gefs,bss-25-gefs,bss-50-gefs,bss-75-gefs'+
                   ',bss-90-gefs,bss-95-gefs,crpss-nodes-gefs,mae-nodes-gefs,rmse-nodes-gefs,bias-nodes-gefs')
    for dom in range(len(lat_dif)):
        tic = time.perf_counter()

        lat_offset = lat_dif[dom]
        lon_offset = lon_dif[dom]
        lat = np.arange(55-lat_offset,55+lat_offset+0.5,0.5)[::-1]
        lon = np.arange(240-lon_offset,240+lon_offset+0.5,0.5)

        era = era_train.sel(latitude=lat,longitude=lon-360)
        if anomaly:
            # taking out all (spatial and temporal) anomaly
            clim = era.groupby("time.dayofyear").mean(dim=["time"])  # clim for each doy for each pixel
            clim_concat = xr.concat([clim.isel(dayofyear=slice(330,366)),clim,clim.isel(dayofyear=slice(0,36))],dim='dayofyear')
            cutoff=0.03
            b, a = signal.butter(5, cutoff, btype='lowpass')
            dUfilt = signal.filtfilt(b, a, clim_concat.gh.values,axis=0)
            dUfilt = dUfilt[36:-36,:,:]
            clim.gh.values = dUfilt

            era = era.groupby("time.dayofyear") - clim
            erav = era_val.sel(latitude=lat,longitude=lon-360).groupby("time.dayofyear") - clim
            gefs_val = gefs_val_full.sel(latitude=lat,longitude=lon-360).groupby("time.dayofyear") - clim
        era = resample_mean(era,'era',res)
        erav = resample_mean(erav,'era',res)
        gefs_val = resample_mean(gefs_val,'gefs',res).sel(step=t_step,method='nearest')

        obs, era = dates_obs_gefs(obs_train, era)
        obs, era = prep_data(era, obs)

        obsv, gefs_val = dates_obs_gefs(obs_val, gefs_val)
        a, erav = dates_obs_gefs(obsv, erav)
        obsv = obsv.Wind.to_numpy()

        if len(obsv) != len(gefs_val.time.values):
            print('something wronggggg with time series length',len(obsv),len(gefs_val.time.values))


        # normalize data (this is actually the z score)
        era_mean = np.mean(era) 
        era_std = np.std(era)
        era = (era - era_mean) / era_std  # normalizing each time frame
        # obs_mean = np.mean(obs) 
        # obs_std = np.std(obs)
        # obs = (obs - obs_mean) / obs_std  # normalizing each time frame

        for (Nx, Ny) in sizes:
            N_nodes = Nx * Ny

            if k_m:  # k means clustering - not done
                kmeans = KMeans(n_clusters=2, random_state=0).fit(era)

                print(kmeans.labels_)

                # the output will be something like:
                #array([0, 0, 0, 1, 1, 1], dtype=int32)
                # the values (0,1) tell you to what cluster does every of your data points correspond to

                #or see were the centres of your clusters are
                kmeans.cluster_centers_
            print('training map...',Nx,'x',Ny)
            with open('stats-'+title+'.txt','a') as file:
                file.write('\n'+str(Nx)+','+str(Ny)+','+str(N_nodes)+','+str(lat_offset)+','+str(lon_offset))

            som = train_som(era)

            #monitor_training(som)

            # map-based statistics
            
            indices = np.arange(N_nodes).reshape(Nx,Ny).T.flatten()
            bmus = BMUs(som)  # the nodes that each forecast belongs to -> use this to look at wind
            distributions = wind_distributions(bmus)
            freq = BMU_frequency(som)  # frequency of each node
            QE = som.QE()  # quantization error
            TE = som.TE()  # topographic error

            #plot_som(Nx, Ny, som.z_raw, indices)

            WSS_nodes = np.empty(N_nodes)
            dist_means = np.empty(N_nodes)
            BSS_nodes = np.empty(N_nodes)

            for i in range(len(distributions)):  # for each node
                mean = np.mean(distributions[i])
                WSS_nodes[i] = np.sum(np.square(distributions[i] - mean))
                dist_means[i] = mean

            # spread of distributions
            dist_spread = np.max(dist_means) - np.min(dist_means)
            dist_std = np.std(dist_means)

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
            with open('stats-'+title+'.txt','a') as file:
                file.write(','+str(TE)+','+str(QE)+','+str(EV)+','+str(PF)+','+str(WSS)
                       +','+str(ks_sig_frac)+','+str(dist_spread)+','+str(dist_std))

            # validation statistics
                

            for ds in [erav, gefs_val]:
                #crps
                crps_som = np.empty((N_nodes,len(obsv)))  # (nodes, forecasts)
                crps_som.fill(np.nan)
                crps_clim = np.empty((N_nodes,len(obsv)))
                crps_clim.fill(np.nan)

                #mae
                mae_som_mean = np.empty((N_nodes,len(obsv)))
                mae_som_mean.fill(np.nan)

                #rmse
                rmse_som_mean = np.empty((N_nodes,len(obsv)))
                rmse_som_mean.fill(np.nan)

                #bias
                bias_som_mean = np.empty((N_nodes,len(obsv)))
                bias_som_mean.fill(np.nan)

                bmus_val = np.zeros(len(obsv), dtype='int')

                for kk, gh in enumerate(ds.gh):  # for each validation date
                    gh = np.reshape(gh.to_numpy(),(gh.latitude.shape[0]*gh.longitude.shape[0]))
                    BMU = np.argmin(np.linalg.norm(gh - som.z_raw, axis=1))
                    #second_bmu = np.argsort(np.linalg.norm(ob - som.z_raw, axis=1))[1]
                    bmus_val[kk] = BMU
                    ob = obsv[kk]

                    crps_som[BMU,kk] =ps.crps_ensemble(ob, distributions[BMU])
                    crps_clim[BMU,kk] =ps.crps_ensemble(ob, obsv)
                    mae_som_mean[BMU,kk] = np.abs(dist_means[BMU] - ob)
                    rmse_som_mean[BMU,kk] = np.square(dist_means[BMU] - ob)
                    bias_som_mean[BMU,kk] = dist_means[BMU] - ob

                # correlation between distribution means and obs through time
                means_val = np.array([dist_means[b] for b in bmus_val]) # time series of the 'predicted' means
                slope, intercept, r_value, p_value, std_err = stats.linregress(means_val, obsv)
                rmse = np.sqrt(np.mean((means_val - obsv)**2))
                mae = np.mean(np.abs(means_val-obsv))
                bias = np.mean(means_val-obsv)

                # ranked continuous ensemble stuff
                # ranking the ensembles (distributions) and seeing if those rankings follow obs
                R_list = []
                crpss_nodes = []
                mae_nodes = []
                rmse_nodes = []
                bias_nodes = []
                for j,d in enumerate(distributions):  # for each distribution
                    crpss_nodes.append(np.nanmean(1-crps_som[j,:]/crps_clim[j,:])) 
                    mae_nodes.append(np.nanmean(mae_som_mean[j,:]))
                    rmse_nodes.append(np.sqrt(np.nanmean(rmse_som_mean[j,:])))
                    bias_nodes.append(np.nanmean(bias_som_mean[j,:]))

                    u_list = []
                    for i,t in enumerate(distributions):  # for each distribution again to compare lol
                        if j == i:
                            continue
                        obs_sorted = np.sort(np.concatenate((d,t)))  # sorting every obs in the distributions
                        F = (np.sum(np.isin(obs_sorted,d).nonzero()[0]) - len(d)*(len(d)+1)/2) / (len(d)*len(t))
                        if F>0.5:
                            u = 1
                        elif F<0.5:
                            u = 0
                        else:
                            u=0.5
                        u_list.append(u)
                    R = 1 + np.sum(u_list)
                    R_list.append(R)

                D = 0.5*(stats.kendalltau(obsv,[R_list[d] for d in bmus_val])[0] + 1)  # generalized discrimination score (Weigel&Mason 2011)

                # windy vs not windy
                splits = [25,50,75,90,95]
                bss = []
                for percentile in splits:
                    split = np.percentile(obs, percentile)
                    target = np.array(obsv > split)  # boolean array of obs
                    dist_cdf = [np.sum(d>split)/len(d) for d in distributions]  # probabilities of being above threshold in each distribution
                    prob = [dist_cdf[d] for d in bmus_val]  # probabilities for the time series
                    prob_clim = np.sum([d>split for d in obs])/len(obs)
                    brier = brier_score_loss(target, prob)
                    brier_clim = brier_score_loss(target, np.repeat(prob_clim,len(target)))
                    bss.append(1 - brier/brier_clim)
                # write things to file
                with open('stats-'+title+'.txt','a') as file:
                    file.write(','+str(r_value)+','+str(rmse)+','+str(mae)+','+str(bias)+','+str(D)+','+str(bss[0])+','+str(bss[1])+','+str(bss[2])+
                            ','+str(bss[3])+','+str(bss[4])+','+str(crpss_nodes)+','+str(mae_nodes)+','+str(rmse_nodes)+','+str(bias_nodes))
            
        toc = time.perf_counter()
        print('Done that domain size',dom+1,f'/3 in {(toc - tic)/60:0.2f} minutes.')
    print('Done',datetime.now())

        
