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
import glob
import pandas as pd



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


def stack_uneven(arrays, fill_value=np.nan):
    # https://stackoverflow.com/questions/58070609/how-to-save-many-np-arrays-of-different-size-in-one-file-eg-one-np-array 
    '''
    Fits arrays into a single numpy array, even if they are
    different sizes. `fill_value` is the default value.

    Args:
            arrays: list of np arrays of various sizes
                (must be same rank, but not necessarily same size)
            fill_value (float, optional):

    Returns:
            np.ndarray
    '''
    sizes = [a.shape for a in arrays]
    max_sizes = np.max(list(zip(*sizes)), -1)
    # The resultant array has stacked on the first dimension
    result = np.full((len(arrays),) + tuple(max_sizes), fill_value)
    for i, a in enumerate(arrays):
      # The shape of this array `a`, turned into slices
      slices = tuple(slice(0,s) for s in sizes[i])
      # Overwrite a block slice of `result` with this array `a`
      result[i][slices] = a
    return result


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

    with open('distributions-'+title,'wb') as f:
        pickle.dump(distributions,f)

    return distributions


def map_stats():
    QE = som.QE()  # quantization error
    TE = som.TE()  # topographic error

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
        
    return None


def gefs_stats():
    global obsv
    #obsv = resample_mean(obsv,'obs',res)  # refill nan spots for indexing (will drop later)
    obsv_loc = obsv.resample(index=str(res)+'H').mean()

    mem = 0
    members = []

    bmus_mems = np.empty((int(len(obsv_loc.index)),len(glob.glob('data/z-gefs/gefs-z-*'))))
    bmus_mems.fill(np.nan)

    for gefs_file in glob.glob('data/z-gefs/gefs-z-*'):
        members.append(int(gefs_file[-5:-3]))
        # open all of them and calculate best bmus, to get memory out of the way
        # save in a dataset so each validation date has a corresponding distribution
        current_gefs = xr.open_dataset(gefs_file).sel(time=val_period_gefs,isobaricInhPa=level)
        current_gefs = current_gefs.shift(time=7)  # now time is really valid time of 7-day forecast -> quick fix, not generalizable >:(
        current_gefs['longitude'] = current_gefs['longitude'] - 360
        if type(t_step) == slice:
            current_gefs = resample_mean(current_gefs,'gefs',res).sel(step=t_step)  # resampled in the 'step' dimension
        else:
            current_gefs = resample_mean(current_gefs,'gefs',res).sel(step=[t_step],method='nearest')  # resampled in the 'step' dimension

        if anomaly:
            current_gefs = current_gefs.sel(latitude=lat,longitude=lon-360).groupby("time.dayofyear") - clim
        else:
            current_gefs = current_gefs.sel(latitude=lat,longitude=lon-360)
        date = slice(obsv_loc.index.values[0],obsv_loc.index.values[-1])
        current_gefs = current_gefs.sel(time=date)  # only keep indices with valid obs

        current_gefs=current_gefs.transpose("time","step","latitude","longitude")  # put time first so that can enumerate thru time
        counter = 0
        for kk, st in enumerate(current_gefs.gh):  # for each validation date
            for ii, gh in enumerate(st):  # for each step
                mem_arr = np.reshape(gh.to_numpy(),(gh.latitude.shape[0]*gh.longitude.shape[0]))
                BMU = np.argmin(np.linalg.norm(mem_arr - som.z_raw, axis=1))
                if np.isnan(gh.any()) or BMU in bad_nodes:  # don't count bad nodes
                    bmus_mems[counter,mem]=np.nan
                else:
                    bmus_mems[counter,mem]=BMU
                counter += 1
        mem += 1
    dist_xr = xr.DataArray(data=bmus_mems,dims=['index','member'],
                           coords=dict(index=obsv_loc.index,member=members))

    # need to make a time series of the distributions
    #obsv_loc=obsv.assign_coords(member=members)
    #obsv_loc['bmu'] = (('index','member'),bmus_mems)
    nan_count = 0
    smth = []

    if res < res_obs:  # do some resampling, avging of shorter res to get longer res forecasts
        res_obs_str = str(res_obs) + 'H'
        obsv_loc = obsv_loc.resample(index=res_obs_str).mean()
    for kk, ob in enumerate(obsv_loc.Wind):  # for each longer period
        dist_list=[]

        if np.isnan(ob.values):
            continue

        # get bmus from this period and add the distributions
        date = pd.to_datetime(ob.index.values)
        if res < res_obs:
            for c in dist_xr.sel(index=str(date.date())).values:  # each date in here will have a list of memberS?
                if np.count_nonzero(np.isnan(c)) > 10:  # if most of the members are bad nodes
                    dist_list.append([np.nan])
                else:
                    dist_list.append(np.concatenate([distributions[int(e)] for e in c if ~np.isnan(e)]))
        else:
            c = dist_xr.sel(index=str(date.date())).values  # each date in here will have a list of memberS?
            if np.count_nonzero(np.isnan(c)) > 10:  # if most of the members are bad nodes
                dist_list.append([np.nan])
            else:
                dist_list.append(np.concatenate([distributions[int(e)] for e in c if ~np.isnan(e)]))
        dist_list = np.concatenate(dist_list)
        if np.isnan(dist_list).all():
            obsv_loc['Wind'][kk] = np.nan
            nan_count += 1
        else:
            smth.append(dist_list)
    # for a, b in enumerate(dist_xr['bmu']): # for each date
    #     if np.count_nonzero(np.isnan(b)) > 10:
    #         obsv_loc['bmu'][a] = np.nan
    #         obsv_loc['Wind'][a] = np.nan
    #         nan_count += 1
    #     else:
    #         smth.append(np.concatenate([distributions[int(c)] for c in b if ~np.isnan(c)]))
    dists_time = stack_uneven(smth)
    obsv_loc = obsv_loc.dropna(dim='index',how='all')

    if len(dists_time) != len(obsv_loc.index):
        print('bad things!!!')
        os.abort()
    discarded = nan_count / (nan_count + len(dists_time))
    
    # remake some of the map stats based on the new distributions
    # pseudo-F and K-S both tell us about the uniqueness of the distributions
    dist_means = np.nanmean(dists_time,axis=1)
    total_mean = np.nanmean(dist_means)
    BSS_nodes = np.empty(len(obsv_loc.index))

    WSS = np.nansum(np.square(dists_time - np.vstack(dist_means)))
    TSS = np.sum(np.square(obs - total_mean))

    sig_count = 0
    m = obs.shape[0]
    for i in range(len(dist_means)):
        BSS_nodes[i] = np.square(dist_means[i] - total_mean)
        # K-S test
        n = dists_time[i,:].shape[0]

        crit = 1.63*np.sqrt((n+m)/(n*m))  # 1.36 is for 95% confidence, 1.07 for 80 1.22 for 90 1.52 for 98 1.63 for 99
        ks = stats.ks_2samp(dists_time[i,:], obs)  # rejection means the distirbutions are different
        if ks.statistic > crit and ks.pvalue < 0.05:  # rejection of null
            sig_count +=1

    ks_sig_frac = sig_count / len(dist_means)  # perentage of nodes that have significantly different distribution
    PF = (np.sum(BSS_nodes)/(len(dist_means)-1)) / (WSS/(m-len(dist_means)))  # pseudo-F statistic

    ####### PERFORMANCE STATS
    #crps
    crps_som = np.empty((len(dist_means)))  # (nodes, forecasts)
    crps_som.fill(np.nan)
    crps_clim = np.empty((len(dist_means)))
    crps_clim.fill(np.nan)

    #mae
    mae_som_mean = np.empty((len(dist_means)))
    mae_som_mean.fill(np.nan)

    #rmse
    rmse_som_mean = np.empty((len(dist_means)))
    rmse_som_mean.fill(np.nan)

    #bias
    bias_som_mean = np.empty((len(dist_means)))
    bias_som_mean.fill(np.nan)
    for kk, gh in enumerate(dist_means):  # for each validation date
        ob = obsv_loc.isel(index=kk).Wind  # wind observation for this date

        crps_som[kk] =ps.crps_ensemble(ob, dists_time[kk,:])
        crps_clim[kk] =ps.crps_ensemble(ob, obs)
        mae_som_mean[kk] = np.abs(gh - ob)
        rmse_som_mean[kk] = np.square(gh - ob)
        bias_som_mean[kk] = gh - ob

    # correlation between distribution means and obs through time
    crpss = 1-crps_som/crps_clim
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(dist_means, obsv_loc.Wind)
    except ValueError:  # throws this error if only 1 node is 'good'
        r_value = np.nan
        p_value = 1
    if p_value > 0.05:
        r_value = np.nan
    rmse = np.sqrt(np.mean((dist_means - obsv_loc.Wind.values)**2))
    mae = np.mean(np.abs(dist_means-obsv_loc.Wind.values))
    bias = np.mean(dist_means-obsv_loc.Wind.values)

    # ranked continuous ensemble stuff
    # ranking the ensembles (distributions) and seeing if those rankings follow obs
    R_list = []

    for j,d in enumerate(dists_time):  # for each distribution
        a = d[~np.isnan(d)]
        u_list = []
        for i,t in enumerate(dists_time):  # for each distribution again to compare lol
            if j == i:
                continue
            b = t[~np.isnan(t)]
            obs_sorted = np.sort(np.concatenate((a,b)))  # sorting every obs in the distributions
            # basically calculating which distribution is bigger
            F = (np.sum(np.isin(obs_sorted,a).nonzero()[0]) - len(a)*(len(a)+1)/2) / (len(a)*len(b))
            if F>0.5:
                u = 1
            elif F<0.5:
                u = 0
            else:
                u=0.5
            u_list.append(u)
        R = 1 + np.sum(u_list)
        R_list.append(R)  # rank of this disitrbution

    D = 0.5*(stats.kendalltau(obsv_loc.Wind.values,R_list)[0] + 1)  # generalized discrimination score (Weigel&Mason 2011)

    # windy vs not windy
    splits = [25,50,75,90,95]
    bss = []
    for percentile in splits:
        split = np.percentile(obs, percentile)
        target = np.array(obsv_loc.Wind > split)  # boolean array of obs
        prob = [np.sum(d>split)/len(d[~np.isnan(d)]) for d in dists_time]  # probabilities of being above threshold in each distribution
        prob_clim = np.sum([d>split for d in obs])/len(obs)
        brier = brier_score_loss(target, prob)
        brier_clim = brier_score_loss(target, np.repeat(prob_clim,len(target)))
        bss.append(1 - brier/brier_clim)
    # write things to file
    with open('stats-'+title+'.txt','a') as file:
        file.write(','+str(PF)+','+str(WSS) +','+str(ks_sig_frac)+','+str(crpss)+','+str(r_value)+
                    ','+str(rmse)+','+str(mae)+','+str(bias)+','+str(D)+','+str(bss[0])+
                    ','+str(bss[1])+','+str(bss[2])+','+str(bss[3])+','+str(bss[4])+','+str(discarded))

    return None


def era_stats():
    crps_som = np.empty((N_nodes,len(obsv.Wind.values)))  # (nodes, forecasts)
    crps_som.fill(np.nan)
    crps_clim = np.empty((N_nodes,len(obsv.Wind.values)))
    crps_clim.fill(np.nan)
    for kk, gh in enumerate(erav.gh):  # for each validation date
        # find the bmu and record the crps for this date
        if np.isnan(gh.any()):
            continue
        else:
            mem_arr = np.reshape(gh.to_numpy(),(gh.latitude.shape[0]*gh.longitude.shape[0]))
            BMU = np.argmin(np.linalg.norm(mem_arr - som.z_raw, axis=1))
            ob = obsv.sel(index=gh.time.values).Wind
            crps_som[BMU,kk] = ps.crps_ensemble(ob, distributions[BMU])
            crps_clim[BMU,kk] =ps.crps_ensemble(ob, obs)

    bad_nodes = []
    for j,d in enumerate(distributions):  # for each node
        # calculate total crpss for each node
        crpss = np.nanmean(1-crps_som[j,:]/crps_clim[j,:])
        if crpss < 0.01:  # flag indices of 'bad' nodes
            bad_nodes.append(j)


    return bad_nodes



if __name__ ==  "__main__":
    # setup
    print('starting',datetime.now())
    x_to_try = np.arange(2,19)
    sizes = []
    for x in x_to_try:
        for y in range(1,x+1):
            if x*y < 37:
                sizes.append((x,y))

    lat_dif = [9,11]  # domain size (degrees on each side of the center)
    lon_dif = [16,20]
    res = 6  # time resolution of map in hours
    res_obs = 6
    t_step = slice(np.array(int(8*24*1e9*60*60),dtype='timedelta64[ns]'),np.array(int(8.9*24*1e9*60*60),dtype='timedelta64[ns]'))
    #t_step = np.array(int(7*24*1e9*60*60),dtype='timedelta64[ns]')

    anomaly = True  # get rid of seasonal anomaly
    train_period=slice("2009-10-01","2020-09-23")
    val_period = slice("2020-10-01","2022-03-31")
    val_period_gefs = slice("2020-09-24","2022-03-24")
    level = 1000

    print('Loading data...')
    obs_full = xr.open_dataset('~/Nextcloud/thesis/bm_cleaned_all.nc')
    obs_full = resample_mean(obs_full,'obs',res)
    obs_full = obs_full.sel(index=obs_full.index.dt.season=='JJA')
    obs_train = obs_full.sel(index=train_period)
    obsv = obs_full.sel(index=val_period)
    obsv = obsv.dropna(dim='index')

    obs_full = None

    title = '6h-summer-1000'

    with open('stats-'+title+'.txt', 'w') as file:
        file.write('Nx,Ny,Nnodes,lat,lon,TE,QE,EV,PF,WSS,KS-frac,range,std,PF-gefs,WSS-gefs,KS-gefs'+
                   ',CRPSS,R-gefs,rmse-gefs,mae-gefs,bias-gefs,D-gefs,bss-25-gefs,bss-50-gefs,bss-75-gefs'+
                   ',bss-90-gefs,bss-95-gefs,frac-discarded')
        
    for dom in range(len(lat_dif)):  # for each domain size
        lat_offset = lat_dif[dom]
        lon_offset = lon_dif[dom]
        lat = np.arange(55-lat_offset,55+lat_offset+0.5,0.5)[::-1]
        lon = np.arange(240-lon_offset,240+lon_offset+0.5,0.5)

        era = xr.open_dataset('era-2009-2022-a.nc').sel(level=level,latitude=lat,longitude=lon-360)
        erav = era.sel(time=val_period)
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
            erav = erav.groupby("time.dayofyear") - clim
        era = resample_mean(era,'era',res)
        erav = resample_mean(erav,'era',res)
        erav = erav.sel(time=obsv.index.values)  # only keep indices with valid obs

        obs, era = dates_obs_gefs(obs_train, era)
        obs, era = prep_data(era, obs)
        #a, erav = dates_obs_gefs(obsv, erav)
        #obsv = obsv.Wind.to_numpy()

        # normalize data (this is actually the z score)
        era_mean = np.mean(era) 
        era_std = np.std(era)
        era = (era - era_mean) / era_std  # normalizing each time frame

        for (Nx, Ny) in sizes:
            N_nodes = Nx * Ny

            print('training map...',Nx,'x',Ny)
            with open('stats-'+title+'.txt','a') as file:
                file.write('\n'+str(Nx)+','+str(Ny)+','+str(N_nodes)+','+str(lat_offset)+','+str(lon_offset))

            som = train_som(era)
            
            indices = np.arange(N_nodes).reshape(Nx,Ny).T.flatten()
            bmus = BMUs(som)  # the nodes that each gh best matches
            distributions = wind_distributions(bmus)
            freq = BMU_frequency(som)  # frequency of each node

            # map-based validation statistics
            map_stats()

            bad_nodes = era_stats()
            
            if len(bad_nodes) == N_nodes:  # if all nodes are bad
                print('all nodes bad')
            else:
                # validation statistics for gefs
                gefs_stats()

        toc = time.perf_counter()
        print('Done that domain size',dom+1,f'/3 in {(toc - tic)/60:0.2f} minutes.')
    print('Done',datetime.now())

        
