import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import xarray as xr
import pandas as pd
from scipy import signal
from resampling import resample_mean
import pickle
from sklearn.metrics import brier_score_loss
import scipy.stats as stats
import properscoring as ps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cmcrameri import cm
import matplotlib.cm as comap
import glob
import os

# testing and plotting

mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['figure.dpi'] = 200
#mpl.rcParams['figure.figsize'] = (5,5)


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


def ens_members(gefs):
    x_axis = pd.date_range('2019-01-01','2019-02-01')
    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0,member=0).wind,linewidth=0.4,c='grey')
    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0,member=1).wind,linewidth=0.4,c='grey')
    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0,member=2).wind,linewidth=0.4,c='grey')
    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0,member=3).wind,linewidth=0.4,c='grey')
    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0,member=4).wind,linewidth=0.4,c='grey')
    

    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0).wind.mean(dim='member'),linewidth=1,c='black')

    lower = gefs.sel(time=x_axis).isel(step=0).wind.min(dim='member')
    upper = gefs.sel(time=x_axis).isel(step=0).wind.max(dim='member')
    plt.fill_between(x_axis, lower, upper, color='cyan', alpha=0.2)

    plt.xticks(rotation = 45)
    plt.ylabel('Wind speed (m/s)')
    plt.show()


def mae_plot(d):
    d = d.sel(member=0)
    x_axis = np.arange(6,15,1)
    mae = []
    a = []
    for v in d.keys():
        if v[0] == 'd':
            mae.append(abs(d[v] - d.Wind).mean())
        elif v[0] == 'c':
            clima.append(abs(d[v] - d.Wind).mean())
    plt.plot(x_axis, mae, label='GEFS control raw')

    plt.plot(x_axis,np.repeat(clima,len(x_axis)),label='Climatology')

    plt.ylabel('MAE Daily avg')
    plt.xlabel('Forecast Day')
    plt.legend()
    plt.show()


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
    cbar.set_label('z500 anomaly (m)')

    plt.suptitle('z500 clusters')
    plt.show()

    return None


def get_climo_det(ds):
    # this makes deterministic climo sorted by hour
    ds = ds.assign_coords(
        {
            #"doy": ds["index.dayofyear"],
            "hour": ds["index.hour"]
        }
    )
    clim_prob = ds  # get hour by   clim_prob.sel(index=clim_prob.index.hour==0)
    clim_det = ds.groupby('index.hour').mean()
    # this disgusting line splits the index into doy and hour, gets average for each hour of each doy
    #o = ds.set_index(index=("doy","hour")).groupby('index').mean().unstack("index")

    # smooth for each hour. not doing cuz no seasonal change.
    # for hi in range(len(climo.hour.values)):
    #     cutoff=0.03  # depends on resolution (ew)
    #     b, a = signal.butter(5, cutoff, btype='lowpass')
    #     dUfilt = signal.filtfilt(b, a, climo.sel(hour=climo.hour.values[hi]).Wind.values,axis=0)
    #     climo.Wind.values[:,hi] = dUfilt
    #climo = climo.mean(dim='doy')

    return clim_prob, clim_det


def era_stats():
    crps_som = np.empty((N_nodes,len(obs_val.Wind.values)))  # (nodes, forecasts)
    crps_som.fill(np.nan)
    crps_clim = np.empty((N_nodes,len(obs_val.Wind.values)))
    crps_clim.fill(np.nan)
    for kk, gh in enumerate(erav.gh):  # for each validation date
        # find the bmu and record the crps for this date
        if np.isnan(gh.any()):
            continue
        else:
            mem_arr = np.reshape(gh.to_numpy(),(gh.latitude.shape[0]*gh.longitude.shape[0]))
            BMU = np.argmin(np.linalg.norm(mem_arr - som.z_raw, axis=1))
            ob = obs_val.sel(index=gh.time.values).Wind
            crps_som[BMU,kk] = ps.crps_ensemble(ob, distributions[BMU])
            crps_clim[BMU,kk] =ps.crps_ensemble(ob, clim_prob.sel(index=clim_prob.index.hour==gh.time.dt.hour).Wind)

    bad_nodes = []
    for j,d in enumerate(distributions):  # for each node
        # calculate total crpss for each node
        crpss = np.nanmean(1-crps_som[j,:]/crps_clim[j,:])
        if crpss < 0.01:  # flag indices of 'bad' nodes
            bad_nodes.append(j)


    return bad_nodes


def forecast():
    do = False
    #obsv = resample_mean(obsv,'obs',res)  # refill nan spots for indexing (will drop later)
    obsv_loc = obs_test.resample(index=str(res)+'H').mean()

    # this is for lead time. if res > 6H, t_step is slice so that we pick up multiple z forecasts
    # need to iterate over lead time inside gefs_file iteration
    t_step = []
    for d in range(6,15): # each day in week 2
        if res == 24:
            t_step.append(slice(np.array(int(d*24*1e9*60*60),dtype='timedelta64[ns]'),
                                np.array(int((d+0.9)*24*1e9*60*60),dtype='timedelta64[ns]')))
        elif res == 6:
            t_step.append(np.array(int(d*24*1e9*60*60),dtype='timedelta64[ns]'))
            t_step.append(np.array(int((d*24+6)*1e9*60*60),dtype='timedelta64[ns]'))
            t_step.append(np.array(int((d*24+12)*1e9*60*60),dtype='timedelta64[ns]'))
            t_step.append(np.array(int((d*24+18)*1e9*60*60),dtype='timedelta64[ns]'))
        else:
            print('only configured for 6h and 24 resolutions right now')
            raise ValueError
        
    mem = 0
    members = []

    bmus_mems = np.empty((int(len(obsv_loc.resample(index='24H').mean().index)),
                          len(glob.glob('data/z-gefs/gefs-z-*')),
                          len(t_step)))
    # bmus mems is shape (days, members, lead times)
    bmus_mems.fill(np.nan)
    if do:
        for gefs_file in glob.glob('data/z-gefs/gefs-z-*'):  # each member
            members.append(int(gefs_file[-5:-3]))
            # open all of them and calculate best bmus, to get memory out of the way
            # save in a dataset so each validation date has a corresponding distribution
            current_gefs = xr.open_dataset(gefs_file).sel(time=test_period_gefs,isobaricInhPa=level)
            current_gefs['longitude'] = current_gefs['longitude'] - 360
            current_gefs = resample_mean(current_gefs,'gefs',6) # resampled in the 'step' dimension
            
            clim_shifted = clim.shift(dayofyear=9) # shift by some days to make clim a bit more lined up time-wise
            clim_shifted.gh.values[:9,:,:] = clim.gh.values[-9:,:,:]
            current_gefs = current_gefs.sel(latitude=lat,longitude=lon-360).groupby("time.dayofyear") - clim_shifted

            for tt,t in enumerate(t_step):  # for each lead time
                # filter gefs by date so that it matches obs date
                if type(t) == slice:
                    gefs_step = current_gefs.sel(step=t)  
                    date = slice(obsv_loc.index.values[0]-t[0],obsv_loc.index.values[-1]-t[0])

                else:
                    gefs_step = current_gefs.sel(step=[t],method='nearest')
                    date = slice(obsv_loc.index.values[0]-t,obsv_loc.index.values[-1]-t)

                gefs_step = gefs_step.sel(time=date)  # only keep indices with valid obs

                # get bmu
                # arrays at this point include nan dates
                # every lead time for every day of every member has a bmu
                gefs_step=gefs_step.transpose("time","step","latitude","longitude")  # put time first so that can enumerate thru
                for kk, st in enumerate(gefs_step.gh):  # for each date
                    for ii, gh in enumerate(st):  # for each step
                        mem_arr = np.reshape(gh.to_numpy(),(gh.latitude.shape[0]*gh.longitude.shape[0]))
                        BMU = np.argmin(np.linalg.norm(mem_arr - som.z_raw, axis=1))
                        if np.isnan(gh.any()) or BMU in bad_nodes:  # don't count bad nodes
                            bmus_mems[kk,mem,tt]=np.nan
                        else:
                            bmus_mems[kk,mem,tt]=BMU
            print('next member')
            mem += 1

        # now iterate by lead time again to get the separate statistics?
        dist_xr = xr.DataArray(data=bmus_mems,dims=['index','member','leadtimes'],
                            coords=dict(index=obsv_loc.resample(index='24H').mean().index,member=members,leadtimes=t_step))
        dist_xr.to_netcdf('forecast-'+title+'.nc')
    else:
        dist_xr = xr.load_dataarray('forecast-'+title+'.nc')



    if res > 6:  # do some resampling, avging of shorter res to get longer res forecasts
        res_obs_str = str(res) + 'H'
        obsv_loc = obsv_loc.resample(index=res_obs_str).mean()

    # get stats for each lead time
    r_list = []
    crps_list = []
    bs50_list = []
    for tt,t in enumerate(t_step):  # for each lead time
        if type(t) == slice:
            t=t[0]
            dist_arr = dist_xr.sel(step=t[0])
        else:
            dist_arr = dist_xr.sel(leadtimes=t)
        nan_count = 0
        smth = []

        for kk, ob in enumerate(obsv_loc.Wind):  # for each testing obs
            dist_list=[]

            if np.isnan(ob.values):  # here we skip nan dates
                continue

            # get bmus from this period and add the distributions
            date = pd.to_datetime(ob.index.values)
            if res > 6:
                for c in dist_arr.sel(index=str(date.date())).values:  # each date in here will have a list of memberS?
                    if np.count_nonzero(np.isnan(c)) > 10:  # if most of the members are bad nodes
                        dist_list.append([np.nan])
                    else:
                        dist_list.append(np.concatenate([distributions[int(e)] for e in c if ~np.isnan(e)]))
            else:
                c = dist_arr.sel(index=str(date.date())).values  # each date in here will have a list of memberS?
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

        dists_time = stack_uneven(smth)
        obsv_cut = obsv_loc.dropna(dim='index',how='all')

        if len(dists_time) != len(obsv_cut.index):
            print('bad things!!!')
            os.abort()
        discarded = nan_count / (nan_count + len(dists_time))
        
        # remake some of the map stats based on the new distributions
        # pseudo-F and K-S both tell us about the uniqueness of the distributions
        dist_means = np.nanmean(dists_time,axis=1)
        total_mean = np.nanmean(dist_means)

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
            ob = obsv_cut.isel(index=kk).Wind  # wind observation for this date
            date = pd.to_datetime(ob.index.values)

            crps_som[kk] =ps.crps_ensemble(ob, dists_time[kk,:])
            crps_clim[kk] =ps.crps_ensemble(ob, clim_prob.sel(index=clim_prob.index.hour==date.hour).Wind)
            mae_som_mean[kk] = np.abs(gh - ob)
            rmse_som_mean[kk] = np.square(gh - ob)
            bias_som_mean[kk] = gh - ob

        # correlation between distribution means and obs through time
        crpss = 1-crps_som/crps_clim
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(dist_means, obsv_cut.Wind)
        except ValueError:  # throws this error if only 1 node is 'good'
            r_value = np.nan
            p_value = 1
        if p_value > 0.05:
            r_value = np.nan
        rmse = np.sqrt(np.mean((dist_means - obsv_cut.Wind.values)**2))
        mae = np.mean(np.abs(dist_means-obsv_cut.Wind.values))
        bias = np.mean(dist_means-obsv_cut.Wind.values)

        # ranked continuous ensemble stuff
        # ranking the ensembles (distributions) and seeing if those rankings follow obs
        # Rank_list = []

        # for j,d in enumerate(dists_time):  # for each distribution
        #     a = d[~np.isnan(d)]
        #     u_list = []
        #     for i,t in enumerate(dists_time):  # for each distribution again to compare lol
        #         if j == i:
        #             continue
        #         b = t[~np.isnan(t)]
        #         obs_sorted = np.sort(np.concatenate((a,b)))  # sorting every obs in the distributions
        #         # basically calculating which distribution is bigger
        #         F = (np.sum(np.isin(obs_sorted,a).nonzero()[0]) - len(a)*(len(a)+1)/2) / (len(a)*len(b))
        #         if F>0.5:
        #             u = 1
        #         elif F<0.5:
        #             u = 0
        #         else:
        #             u=0.5
        #         u_list.append(u)
        #     R = 1 + np.sum(u_list)
        #     Rank_list.append(R)  # rank of this disitrbution

        # D = 0.5*(stats.kendalltau(obsv_loc.Wind.values,Rank_list)[0] + 1)  # generalized discrimination score (Weigel&Mason 2011)

        # windy vs not windy
        splits = [50]
        bss = []
        for percentile in splits:
            split = np.percentile(obs_train, percentile)
            target = np.array(obsv_loc.Wind > split)  # boolean array of obs
            prob = [np.sum(d>split)/len(d[~np.isnan(d)]) for d in dists_time]  # probabilities of being above threshold in each distribution
            prob_clim = np.sum([d>split for d in obs_train])/len(obs_train)
            brier = brier_score_loss(target, prob)
            brier_clim = brier_score_loss(target, np.repeat(prob_clim,len(target)))
            bss.append(1 - brier/brier_clim)
        bs50 = brier
        # add to lists
        r_list.append(r_value)
        crps_list.append(np.nanmean(crps_som))
        bs50_list.append(bs50)
    return r_list, crps_list, bs50_list



if __name__ == '__main__':
    res = 6 # time resolution in hours
    train_period=slice("2009-10-01","2020-09-23")
    val_period = slice("2020-10-01","2022-03-31")
    test_period_gefs = slice("2022-03-25","2024-03-25")  # 6 days earlier so that dates line up
    test_period = slice("2022-04-01","2024-04-01")
    seas = 'JJA'
    level = 1000

    # setup
    Nx = 8
    Ny = 2
    N_nodes = Nx * Ny
    title = '6h-8x2-summer'
    lat_offset = 11
    lon_offset = 20
    lat = np.arange(55-lat_offset,55+lat_offset+0.5,0.5)[::-1]
    lon = np.arange(240-lon_offset,240+lon_offset+0.5,0.5)

    # obs data
    print('Prepping data...')
    obs_full = xr.open_dataset('~/Nextcloud/thesis/bm_cleaned_all.nc')
    obs_full = resample_mean(obs_full,'obs',res)
    obs_full = obs_full.sel(index=obs_full.index.dt.season==seas)

    # use training obs for climatology, val obs for finding bad nodes, testing obs for testing
    obs_train = obs_full.sel(index=train_period)
    clim_prob, clim_det = get_climo_det(obs_train)  # for both datasets need to select the intended hour
    obs_val = obs_full.sel(index=val_period)
    obs_val = obs_val.dropna(dim='index')
    obs_test = obs_full.sel(index=test_period)
    obs_test = obs_test.dropna(dim='index')
    obs_full = None  # free up memory

    # era data (used for validation - finding bad nodes)
    era = xr.open_dataset('era-2009-2022-a.nc').sel(level=level,latitude=lat,longitude=lon-360)
    erav = era.sel(time=val_period)
    print('Processing data...')
    # taking out all (spatial and temporal) anomaly
    clim = era.groupby("time.dayofyear").mean(dim=["time"])  # clim for each doy for each pixel
    clim_concat = xr.concat([clim.isel(dayofyear=slice(330,366)),clim,clim.isel(dayofyear=slice(0,36))],dim='dayofyear')
    cutoff=0.03
    b, a = signal.butter(5, cutoff, btype='lowpass')
    dUfilt = signal.filtfilt(b, a, clim_concat.gh.values,axis=0)
    dUfilt = dUfilt[36:-36,:,:]
    clim.gh.values = dUfilt

    erav = erav.groupby("time.dayofyear") - clim
    erav = resample_mean(erav,'era',res)
    erav = erav.sel(time=obs_val.index.values)  # only keep indices with valid obs

    # open som and dists
    print('Opening map and doing forecast')
    with open('trained-map-'+title,'rb') as handle:
        som = pickle.load(handle)
    indices = np.arange(N_nodes).reshape(Nx,Ny).T.flatten()
    #plot_som(Nx, Ny, som.z_raw, indices)

    with open('distributions-'+title,'rb') as handle:
        distributions = pickle.load(handle)


    # do forecasting, stats
    bad_nodes = era_stats()

    forecast()


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

    
