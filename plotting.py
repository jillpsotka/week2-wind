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
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import math
from seaborn import violinplot as violin

# testing and plotting

mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['figure.dpi'] = 250
mpl.rcParams['image.cmap'] = 'cmc.lipari'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#0072B2","#CC79A7","#a18252","#4f6591"])


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



def plot_som(Nx, Ny, z, indices):
    proj=ccrs.PlateCarree()
    vmin = np.min(z)
    vmax = np.max(z)  # colorbar range
    fig, axes = plt.subplots(nrows=Ny, ncols=Nx,sharex=True,sharey='row',layout='constrained',figsize=(Nx*2,Ny*2),
                             subplot_kw={'projection': proj, 'aspect':1.7},gridspec_kw = {'wspace':0.1, 'hspace':0.08})
    im = comap.ScalarMappable(norm=Normalize(vmin,vmax),cmap='cmc.lipari')
    for kk, ax in enumerate(axes.flatten()):
        var = z[indices[kk],:].reshape(45,81)
        ax.set_extent(([219,261,43.25,65.25]))
        ax.set_title('Node '+str(indices[kk]))      
        
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_0_boundary_lines_land', scale='50m', facecolor='none', edgecolor='k',alpha=0.6,linewidth=0.6))
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_1_states_provinces_lines', scale='50m',facecolor='none', edgecolor='k',alpha=0.6,linewidth=0.6))
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', \
            scale='50m', edgecolor='k', facecolor='none', alpha=0.6,linewidth=0.5))
        
        cs = ax.contourf(lon, lat, var, vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),cmap='cmc.lipari')
        ax.scatter(360-120.4306,55.6986,c='yellow',transform=ccrs.PlateCarree(),s=5,marker='*',zorder=20)

        # Create gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.8, color='black', alpha=0.2,linestyle='--')
        # Manipulate gridlines number and spaces
        gl.ylocator = mticker.FixedLocator(np.arange(-90,90,10))
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10))
        gl.xlabel_style = {'size': 7,'rotation':40}
        gl.ylabel_style = {'size': 7} 
        if kk > (Ny*Nx) - Nx - 1:
            gl.bottom_labels = False
        if kk % Nx == 0:
            gl.left_labels = False

    #cbar_ax = fig.add_axes([0.05, 0.07, 0.45*Nx, 0.03])
    cbar = fig.colorbar(im,ax=axes,fraction=0.046, pad=0.06,orientation='horizontal')
    cbar.set_label('100 kPa Geopotential Height Anomaly (m)',size=12)

    #plt.suptitle('z500 clusters')
    plt.savefig('plots/som-summer-8x2.pdf')

    return None

 
def plot_distributions(distributions):
        
    fig, axes = plt.subplots(nrows=Ny, ncols=Nx, figsize=(Nx*2,Ny*1.9),sharex=True,sharey=True,
                             gridspec_kw = {'wspace':0.3, 'hspace':0.2})
    vmin = np.nanmin(obs_train.Wind.values)
    vmax = np.nanmax(obs_train.Wind.values)
    means = []
    
    for i, ax in enumerate(axes.flatten()):
        distribution = distributions[indices[i]]  # wind obs that belong to this node -> this is our distribution
        ax.hist(distribution, range=(vmin,vmax),color='black',
                weights=np.zeros_like(distribution) + 1. / distribution.size)
        dist_mean = np.mean(distribution)
        means.append(dist_mean)
        ind = int((dist_mean-3)*(255)/(9-3)) # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        col_ar = cm.lipari.colors[ind]
        col_tuple = tuple(col_ar)
        ax.patch.set_facecolor(col_tuple)
        ax.patch.set_alpha(0.9)
        distributions.append(distribution)
        ax.tick_params(axis='both',labelsize=7)
        if i > (Ny*Nx) - Nx - 1:
            ax.set_xlabel('(m/s)',fontsize=8)
        if i % Nx == 0:
            ax.set_ylabel('Frequency',fontsize=8)

    im = comap.ScalarMappable(norm=Normalize(np.min(means),np.max(means)),cmap='cmc.lipari')

    cbar = fig.colorbar(im,ax=axes,fraction=0.046, pad=0.17,orientation='horizontal')
    cbar.set_label('Mean Wind Speed (m/s)',size=9)
    plt.show()

    plt.close(fig)
    return None


def plot_domain():
    proj=ccrs.PlateCarree()
    fig, ax = plt.subplots(nrows=1, ncols=1,layout='constrained',figsize=(10,10),
                             subplot_kw={'projection': proj, 'aspect':1.7},gridspec_kw = {'wspace':0.1, 'hspace':0.08})
    sizes_lat = [14,18,22,26]
    sizes_lon = [24,32,40,48]
    central_lat =55
    central_lon =240

    ax.set_extent(([202,305,38,67]))
    
    ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
        name='admin_0_boundary_lines_land', scale='50m', facecolor='none', edgecolor='k',alpha=0.6,linewidth=0.7))
    ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
        name='admin_1_states_provinces_lines', scale='50m',facecolor='none', edgecolor='k',alpha=0.6,linewidth=0.7))
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', \
        scale='50m', edgecolor='k', facecolor='none', alpha=0.6,linewidth=0.7))
    for s in range(len(sizes_lat)):
        lat_min = central_lat - sizes_lat[s] / 2
        lat_max = central_lat + sizes_lat[s] / 2
        lon_min = central_lon - sizes_lon[s] / 2
        lon_max = central_lon + sizes_lon[s] / 2
        lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
        lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
        ax.plot(lons, lats, transform=ccrs.PlateCarree())
    #ax.scatter(360-120.4306,55.6986,c='yellow',transform=ccrs.PlateCarree(),s=5,marker='*',zorder=20)

    # Create gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.8, color='black', alpha=0.2,linestyle='--')
    # Manipulate gridlines number and spaces
    gl.ylocator = mticker.FixedLocator(np.arange(-90,90,10))
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10))
    gl.xlabel_style = {'size': 14,'rotation':40}
    gl.ylabel_style = {'size': 14}
    gl.bottom_labels = True
    gl.left_labels = True

    #plt.suptitle('z500 clusters')
    plt.savefig('plots/domains.pdf')



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


def era_stats(aggression):
    if aggression == 0:
        thresh = -10
    elif aggression == 1:
        thresh = 0.01
    elif aggression == 2:
        thresh = 0.05
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
        if crpss < thresh:  # flag indices of 'bad' nodes
            bad_nodes.append(j)


    return bad_nodes


def forecast():
    if res > 6:
        leads = []
        for l in t_step:
            leads.append(l.start)
            end = l.start + np.timedelta64(6,'h')
            while end < l.start+np.timedelta64(res,'h'):
                leads.append(end)
                end += np.timedelta64(6,'h')
    else:
        leads = t_step

    if do:
        files = glob.glob('/users/jpsotka/Nextcloud/z-gefs/gefs-z-*')
        time_axis = pd.date_range(test_period_gefs.start,test_period_gefs.stop,freq=res_str)
        bmus_mems = np.empty((len(time_axis),len(files),len(leads)))
        # bmus_mems = np.empty((int(len(obsv_loc.resample(index='24H').mean().index)),
        #                   len(files),len(leads)))
        # bmus mems is shape (days, members, lead times)
        bmus_mems.fill(np.nan)

        dist_xr = xr.DataArray(data=bmus_mems,dims=['index','member','leadtime'],
            coords=dict(index=time_axis,member=range(1,len(files)+1),leadtime=leads))
        
        for gefs_file in files:  # each member
            mem=int(gefs_file[-5:-3])
            print('member ',mem)
            # open all of them and calculate best bmus, to get memory out of the way
            # save in a dataset so each validation date has a corresponding distribution
            current_gefs = xr.open_dataset(gefs_file).sel(time=test_period_gefs,isobaricInhPa=level)
            current_gefs['longitude'] = current_gefs['longitude'] - 360
            current_gefs = resample_mean(current_gefs.sel(latitude=lat,longitude=lon-360),'gefs',6) # resampled in the 'step' dimension

            for tt,t in enumerate(t_step):  # for each lead time
                # filter gefs by date so that it matches obs date
                if type(t) == slice:
                    gefs_step = current_gefs.sel(step=t)  
                    date = slice(obs_test.index.values[0]-t.start,obs_test.index.values[-1]-t.start)
                    t = t.start

                else:
                    gefs_step = current_gefs.sel(step=[t],method='nearest')
                    date = slice(obs_test.index.values[0]-t,obs_test.index.values[-1]-t)

                gefs_step = gefs_step.sel(time=date)  # only keep indices with valid obs

                # get bmu
                # arrays at this point include nan dates
                # every lead time for every day of every member has a bmu
                gefs_step=gefs_step.transpose("time","step","latitude","longitude")  # put time first so that can enumerate thru
                for kk, st in enumerate(gefs_step):  # for each date
                    date = st.time.values
                    if date + t in obs_test.index.values:
                        st = st - clim.sel(dayofyear=pd.to_datetime(date+t).day_of_year)  # ANOMOLAY STUFF
                        for ii, gh in enumerate(st.gh):  # for each step
                            mem_arr = np.reshape(gh.to_numpy(),(gh.latitude.shape[0]*gh.longitude.shape[0]))
                            BMU = np.argmin(np.linalg.norm(mem_arr - som.z_raw, axis=1))
                            l = leads[int(tt*(res/6)+ii)]
                            if np.isnan(gh.any()) or BMU in bad_nodes:  # don't count bad nodes
                                dist_xr.loc[dict(index=date,member=mem,leadtime=l)] = np.nan
                            else:
                                dist_xr.loc[dict(index=date,member=mem,leadtime=l)] = BMU
                                #bmus_mems[kk,mem,int(tt*(res/6)+ii)]=BMU
        dist_xr = dist_xr.dropna(dim='index',how='all')

        dist_xr.to_netcdf('forecast-'+title+'.nc')
    else:
        dist_xr = xr.load_dataarray('forecast-'+title+'.nc')

    # get stats for each lead time
    r_list = []
    crps_list = []
    bs50_list = []
    mae_list = []
    bias_list = []
    discarded = []

    bmus_mems = np.empty((len(obs_test.index),len(t_step_plot)))
    bmus_mems.fill(np.nan)
    stats_xr = xr.Dataset(data_vars=dict(crps=(['index','leadtime'],bmus_mems),bias=(['index','leadtime'],bmus_mems.copy()),
                                         prob=(['index','leadtime'],bmus_mems.copy()),width=(['index','leadtime'],bmus_mems.copy()),
                                         width90=(['index','leadtime'],bmus_mems.copy()),width50=(['index','leadtime'],bmus_mems.copy())),
            coords=dict(index=obs_test.index,leadtime=t_step_plot))
    for tt,t in enumerate(t_step):  # for each lead time
        obsv_loc = obs_test.copy(deep=True)
        dist_arr = dist_xr.sel(leadtime=t)
        nan_count = 0
        smth = []

        for kk, ob in enumerate(obsv_loc.Wind):  # for each testing obs
            dist_list=[]

            if np.isnan(ob.values):  # here we skip nan obs
                continue

            # get bmus from this period and add the distributions
            date = ob.index.values
            if res > 6:
                if (date-t.start) in dist_arr.index.values:
                    for c in dist_arr.sel(index=date-t.start).values.T.squeeze():  # each date in here will have a list of memberS?
                        if np.count_nonzero(np.isnan(c)) > 0.5*len(dist_xr.member):  # if many of the members are bad nodes
                            dist_list.append([np.nan])
                        else:
                            dist_list.append(np.concatenate([distributions[int(e)] for e in c if ~np.isnan(e)]))
                else:
                    continue
                dis_count = 0
                for l in dist_list:
                    if np.isnan(l[0]):
                        dis_count += 1
                if dis_count > 0.5*len(dist_list):
                    dist_list = [np.array([np.nan])]

            else:
                if (date-t) in dist_arr.index.values:
                    c = dist_arr.sel(index=date-t).values
                    if np.count_nonzero(np.isnan(c)) > 0.5*len(dist_xr.member):  # if most of the members are bad nodes
                        dist_list.append([np.nan])
                    else:
                        dist_list.append(np.concatenate([distributions[int(e)] for e in c if ~np.isnan(e)]))
                else:
                    continue
            
            dist_list = np.concatenate(dist_list)
            if np.isnan(dist_list).all():
                obsv_loc['Wind'][kk] = np.nan
                nan_count += 1
            else:
                smth.append(dist_list)

        dists_time = stack_uneven(smth)
        obsv_cut = obsv_loc.dropna(dim='index',how='all')
        if res == 6:
            obsv_cut = obsv_cut.where(obsv_cut['index.hour']==pd.to_datetime(t.astype('datetime64')).hour, drop=True)
        elif res == 12:
            obsv_cut = obsv_cut.where(obsv_cut['index.hour']==pd.to_datetime(t.start.astype('datetime64')).hour, drop=True)

        if len(dists_time) != len(obsv_cut.index):
            print('bad things!!!')
            raise ValueError
            os.abort()
        discarded.append(nan_count / (nan_count + len(dists_time)))
        
        # remake some of the map stats based on the new distributions
        # pseudo-F and K-S both tell us about the uniqueness of the distributions
        dist_means = np.nanmean(dists_time,axis=1)
        total_mean = np.nanmean(dist_means)

        ####### PERFORMANCE STATS
        #crps
        crps_som = np.empty((len(dist_means)))  # (nodes, forecasts)
        crps_som.fill(np.nan)
        if res > 6:
            t = t.start

        for kk, gh in enumerate(dist_means):  # for each validation date
            ob = obsv_cut.isel(index=kk).Wind  # wind observation for this date
            date = pd.to_datetime(ob.index.values)

            crps_som[kk] =ps.crps_ensemble(ob.values, dists_time[kk,:])
            stats_xr['crps'].loc[dict(index=date,leadtime=t)]=ps.crps_ensemble(ob.values, dists_time[kk,:])
            stats_xr['bias'].loc[dict(index=date,leadtime=t)]= dist_means[kk]-ob.values

            dists_nonan = dists_time[kk,:][~np.isnan(dists_time[kk,:])]
            stats_xr['prob'].loc[dict(index=date,leadtime=t)]= sum(v < ob.values for v in dists_nonan) / len(dists_nonan)

            stats_xr['width'].loc[dict(index=date,leadtime=t)]= np.max(dists_nonan) - np.min(dists_nonan)
            w95=np.percentile(dists_nonan, 95)
            w5=np.percentile(dists_nonan, 5)
            w75=np.percentile(dists_nonan, 75)
            w25=np.percentile(dists_nonan, 25)
            dists_90 = dists_nonan[np.where(np.logical_and(dists_nonan>=w5, dists_nonan<=w95))]
            dists_50 = dists_nonan[np.where(np.logical_and(dists_nonan>=w25, dists_nonan<=w75))]
            stats_xr['width90'].loc[dict(index=date,leadtime=t)]= np.max(dists_90) - np.min(dists_90)
            stats_xr['width50'].loc[dict(index=date,leadtime=t)]= np.max(dists_50) - np.min(dists_50)

        # correlation between distribution means and obs through time
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
        for percentile in splits:
            split = np.percentile(obs_train.Wind.dropna(dim='index'), percentile)
            target = np.array(obsv_cut.Wind > split)  # boolean array of obs
            prob = [np.sum(d>split)/len(d[~np.isnan(d)]) for d in dists_time]  # probabilities of being above threshold in each distribution
            brier = brier_score_loss(target, prob)
        bs50 = brier
        # add to lists
        r_list.append(r_value)
        crps_list.append(crps_som)
        bs50_list.append(bs50)
        bias_list.append(bias)
        mae_list.append(mae)

    stats_xr['discarded'] = (['leadtime'],discarded)
    stats_xr['r'] = (['leadtime'],r_list)
    stats_xr['mae'] = (['leadtime'],mae_list)
    stats_xr['bs50'] = (['leadtime'],bs50_list)
    return stats_xr


def clim_stats(ds):
    obs = obs_test.dropna(dim='index')

    nan_arr = np.repeat(np.nan,len(obs.index))
    obs['crps_clim'] = (['index'],nan_arr.copy())
    obs['bias_clim'] = (['index'],nan_arr.copy())
    obs['mae_clim'] = (['index'],nan_arr.copy())

    ds['crps_clim'] = (['index'],np.repeat(np.nan,len(obs_test.index)))
    ds['mae_clim'] = (['index'],np.repeat(np.nan,len(obs_test.index)))
    ds['bias_clim'] = (['index'],np.repeat(np.nan,len(obs_test.index)))
    ds['obs'] = (['index'],obs_test.Wind.values)
    ds['prob_clim'] = (['index'],np.repeat(np.nan,len(obs_test.index)))
    ds['width_clim'] = (['index'],np.repeat(np.nan,len(obs_test.index)))
    ds['width50_clim'] = (['index'],np.repeat(np.nan,len(obs_test.index)))
    ds['width90_clim'] = (['index'],np.repeat(np.nan,len(obs_test.index)))

    for kk, gh in enumerate(obs.index):  # for each obs
        ob = obs.sel(index=gh).Wind.values  # wind observation for this date
        date = pd.to_datetime(gh.values)
        obs['crps_clim'].loc[gh] = ps.crps_ensemble(ob, clim_prob.sel(index=clim_prob.index.hour==date.hour).Wind)
        obs['mae_clim'].loc[gh] = np.abs(clim_det.sel(hour=date.hour).Wind.values-ob)
        obs['bias_clim'].loc[gh] = clim_det.sel(hour=date.hour).Wind.values-ob

        ds['crps_clim'].loc[dict(index=gh)] = ps.crps_ensemble(ob, clim_prob.sel(index=clim_prob.index.hour==date.hour).Wind)
        ds['mae_clim'].loc[dict(index=gh)] = np.abs(clim_det.sel(hour=date.hour).Wind.values-ob)
        ds['bias_clim'].loc[dict(index=gh)] = clim_det.sel(hour=date.hour).Wind.values-ob

        dists_nonan = clim_prob.sel(index=clim_prob.index.hour==date.hour).Wind
        dists_nonan = dists_nonan[~np.isnan(dists_nonan)]
        #ds['prob_clim'].loc[dict(index=gh)] = sum(v < ob for v in dists_nonan) / len(dists_nonan)

        ds['width_clim'].loc[dict(index=gh)]= np.max(dists_nonan) - np.min(dists_nonan)
        w95=np.percentile(dists_nonan, 95)
        w5=np.percentile(dists_nonan, 5)
        w75=np.percentile(dists_nonan, 75)
        w25=np.percentile(dists_nonan, 25)
        dists_90 = dists_nonan[np.where(np.logical_and(dists_nonan>=w5, dists_nonan<=w95))]
        dists_50 = dists_nonan[np.where(np.logical_and(dists_nonan>=w25, dists_nonan<=w75))]
        ds['width90_clim'].loc[dict(index=gh)]= np.max(dists_90) - np.min(dists_90)
        ds['width50_clim'].loc[dict(index=gh)]= np.max(dists_50) - np.min(dists_50)

    if res < 24:  # do stack-y things for lead times
        mae_clim = obs['mae_clim'].groupby('index.hour').mean()
        mae_clim = np.tile(mae_clim,9)  # tile for 9 days
        bias_clim = obs['bias_clim'].groupby('index.hour').mean()
        bias_clim = np.tile(bias_clim,9)  # tile for 9 days
        crps_clim = obs['crps_clim'].groupby('index.hour').mean()
        crps_clim = np.tile(crps_clim,9)  # tile for 9 days
    else:
        mae_clim = np.repeat(np.nanmean(obs['mae_clim']),len(t_step))
        bias_clim = np.repeat(np.nanmean(obs['bias_clim']),len(t_step))
        crps_clim = np.repeat(np.nanmean(obs['crps_clim']),len(t_step))

    # windy vs not windy
    splits = [50]
    bss = []
    for percentile in splits:
        split = np.percentile(obs_train.Wind.dropna(dim='index'), percentile)
        target = np.array(obs.Wind > split)  # boolean array of obs
        prob_clim = np.sum([d>split for d in obs_train.Wind.dropna(dim='index')])/len(obs_train.Wind.dropna(dim='index'))
        brier_clim = brier_score_loss(target, np.repeat(prob_clim,len(target)))  # should be 0.25
    ds['bs50_clim'] = brier_clim
    bs50_clim = np.repeat(brier_clim, len(t_step))

    return ds


def gefs_stats(ds, st):
    obs = obs_test.dropna(dim='index')

    bmus_mems = np.empty((len(obs_test.index),len(t_step_plot)))
    bmus_mems.fill(np.nan)
    st['crps_gefs'] = (['index','leadtime'],bmus_mems)
    st['bias_gefs'] = (['index','leadtime'],bmus_mems.copy())
    st['prob_gefs'] = (['index','leadtime'],bmus_mems.copy())
    st['width_gefs'] = (['index','leadtime'],bmus_mems.copy())
    st['width90_gefs'] = (['index','leadtime'],bmus_mems.copy())
    st['width50_gefs'] = (['index','leadtime'],bmus_mems.copy())


    mae_list = []
    bias_list = []
    r_list = []
    crps_list = []
    bs50_list = []

    for tt,t in enumerate(t_step):  # for each lead time
        # filter gefs by date so that it matches obs date
        if type(t) == slice:
            gefs_step = ds.sel(step=[t.start],method='nearest')  # this probably wrong for 24hr stuff because dont need to do fancy things for wind
            t = t.start
        else:
            gefs_step = ds.sel(step=[t],method='nearest')

        # arrays at this point include nan dates
        gefs_step=gefs_step.transpose("time","step","member")  # put time first so that can enumerate thru
        mae = []
        bias = []
        crps = []
        g_time_series = []
        g_time_series_members = []
        ob_time_series = []

        for kk, gh in enumerate(gefs_step):  # for each date
            date = pd.to_datetime(gh.time.values)
            obs_date = date + t
            if obs_date in obs.index.values:
                ob = obs.sel(index=obs_date).Wind.values
                forecast = gh.mean(dim='step').values

                crps.append(ps.crps_ensemble(ob, forecast))
                st['crps_gefs'].loc[dict(index=obs_date,leadtime=t)] = ps.crps_ensemble(ob, forecast)

                g_time_series_members.append(forecast)
                gh = gh.mean().values
                g_time_series.append(gh)
                ob_time_series.append(ob)
                mae.append(np.abs(gh-ob))
                bias.append(gh - ob)
                st['bias_gefs'].loc[dict(index=obs_date,leadtime=t)] = gh - ob

                dists_nonan = forecast[~np.isnan(forecast)]
                st['prob_gefs'].loc[dict(index=obs_date,leadtime=t)]= sum(v < ob for v in dists_nonan) / len(dists_nonan)

                st['width_gefs'].loc[dict(index=obs_date,leadtime=t)]= np.max(dists_nonan) - np.min(dists_nonan)
                w95=np.percentile(dists_nonan, 95)
                w5=np.percentile(dists_nonan, 5)
                w75=np.percentile(dists_nonan, 75)
                w25=np.percentile(dists_nonan, 25)
                dists_90 = dists_nonan[np.where(np.logical_and(dists_nonan>=w5, dists_nonan<=w95))]
                dists_50 = dists_nonan[np.where(np.logical_and(dists_nonan>=w25, dists_nonan<=w75))]
                st['width90_gefs'].loc[dict(index=obs_date,leadtime=t)]= np.max(dists_90) - np.min(dists_90)
                st['width50_gefs'].loc[dict(index=obs_date,leadtime=t)]= np.max(dists_50) - np.min(dists_50)
        mae_list.append(np.mean(mae))
        bias_list.append(np.mean(bias))
        crps_list.append(np.array(crps))

        slope, intercept, r_gefs, p_gefs, std_err = stats.linregress(g_time_series, ob_time_series)
        if p_gefs> 0.05:
            r_gefs = np.nan
        r_list.append(r_gefs)
        

        splits = [50]
        bss = []
        for percentile in splits:
            split = np.percentile(obs_train.Wind.dropna(dim='index'), percentile)
            target = np.array(ob_time_series > split)  # boolean array of obs
            prob = [np.sum(d>split)/len(d[~np.isnan(d)]) for d in g_time_series_members]  # probabilities of being above threshold in each distribution
            brier = brier_score_loss(target, prob)
        bs50_list.append(brier)

    st['r_gefs'] = (['leadtime'],r_list)
    st['mae_gefs'] = (['leadtime'],mae_list)
    st['bs50_gefs'] = (['leadtime'],bs50_list)

    return st
            


def plot_error(som_error, clim_error, gefs_error, ltimes):
    ltimes = np.array(ltimes,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)

    fig, ax = plt.subplots(figsize=(7,4))

    ax.plot(ltimes, som_error, label='SOM',zorder=3,marker='o',markersize=3)
    ax.plot(ltimes, gefs_error, label='GEFS 80m wind',zorder=2,marker='o',markersize=3)
    ax.plot(ltimes, clim_error, label='Climatology',c='black',zorder=1)
    ax.set_xlabel('Lead time (days)')
    ax.set_ylabel('CRPS (m/s)')
    ax.set_xticks(range(6,16))
    ax.xaxis.grid(True,which='both',linestyle='--')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return None

def plot_rel_error(som_error, clim_error, gefs_error, ltimes):
    som_error = np.array(som_error)
    gefs_error = np.array(gefs_error)
    clim_error = np.array(clim_error)
    ltimes = np.array(ltimes,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(ltimes, np.zeros(len(ltimes)),c='black',linewidth=1,zorder=1)
    ax.plot(ltimes,1-som_error/clim_error,label='SOM',zorder=3)
    ax.plot(ltimes,1-gefs_error/clim_error,label='GEFS 80m wind',zorder=2)
    ax.set_xlabel('Lead time (days)')
    ax.set_ylabel('CRPSS')
    ax.set_xticks(range(6,16))
    ax.xaxis.grid(True,which='both',linestyle='--')
    ax.legend(fontsize=12)
    plt.tight_layout()

    return None


def box_plots(som_error,gefs_error, ltimes):
    ltimes = np.array(ltimes,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.axhline(y=0,c='black',linewidth=1,zorder=1)
    ax.set_xlabel('Lead time (days)')
    ax.set_ylabel('CRPSS')

    for l in range(len(som_error.leadtime)):
        ax.boxplot(som_error.isel(leadtime=l).dropna(dim='index'),positions=[ltimes[l]],patch_artist=True,showfliers=False,medianprops=dict(color='red'))
        ax.boxplot(gefs_error.isel(leadtime=l).dropna(dim='index'),positions=[ltimes[l]+0.18],patch_artist=True,showfliers=False,boxprops=dict(facecolor='pink'),medianprops=dict(color='red'))


    ax.set_xticks(range(6,16))
    ax.set_xticklabels(range(6,16))

    plt.show()

    print('Done')


def box_plots_6(som_error, gefs_error, ltimes):
    ltimes = np.array(ltimes,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)
    ml = mticker.MultipleLocator(0.5)

    fig, axes = plt.subplots(4,1,figsize=(5,6),sharex=True,gridspec_kw = {'hspace':0.2})

    for i , ax in enumerate(axes):
        ax.axhline(y=0,c='black',linewidth=1,zorder=1)
        if i == 3:
            ax.set_xlabel('Lead time (days)')
        ax.set_ylabel('CRPSS')

        for l in range(int(len(som_error.leadtime)/4)):
            ax.boxplot(som_error.isel(leadtime=(l*4+i)).dropna(dim='index'),positions=[ltimes[l*4+i]-0.1],patch_artist=True,showfliers=False,medianprops=dict(color='red'))
            ax.boxplot(gefs_error.isel(leadtime=(l*4+i)).dropna(dim='index'),positions=[ltimes[l*4+i]+0.1],patch_artist=True,showfliers=False,boxprops=dict(facecolor='pink'),medianprops=dict(color='red'))

        ax.set_xticks(range(6,16))
        ax.set_xticklabels(range(6,16))
        ax.set_yticks(np.arange(-1,1.1,1))
        ax.set_yticklabels(np.arange(-1,1.1,1))
        ax.set_ylim(bottom=-2)
        ax.yaxis.set_minor_locator(ml)

    plt.show()

    print('Done')


def violin_plots(ds, ltimes):
    ltimes = np.array(ltimes,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)

    ds = ds.to_dataframe()

    ds = pd.melt(ds,value_vars=['crpss','crpss_gefs'],ignore_index=False).reset_index()

    violin(data=ds,x='leadtime',hue='variable',y='value',split=True,gap=0.1,inner='quart')


    plt.show()

    print('Done')


def plot_discarded(ds,ltimes):
    ltimes = np.array(ltimes,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(ltimes, ds,c='black',linewidth=1,zorder=1)
    ax.set_xlabel('Lead time (days)')
    ax.set_ylabel('Fraction of discarded forecasts')
    ax.set_xticks(range(6,16))
    ax.xaxis.grid(True,which='both',linestyle='--')
    plt.tight_layout()
    #plt.savefig('plots/crps-summer-6h-discarded.png')


def pit(ds,ltimes):
    ltimes = np.array(ltimes,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)

    fig, axes = plt.subplots(nrows=3, ncols=math.ceil(len(ltimes)/3), figsize=(7,5),sharey=True,
                             gridspec_kw = {'wspace':0.3, 'hspace':0.7})
    
    for i, ax in enumerate(axes.flatten()):
        if i < len(ltimes):
            probs = ds['prob'].isel(leadtime=i)[~np.isnan(ds['prob'].isel(leadtime=i))]
            
            ax.hist(probs,bins=np.arange(0,1.1,0.1),weights=(np.zeros_like(probs) + 1./probs.size)*100,edgecolor='black')
            ax.set_xticks(np.arange(0,1.1,0.1))
            ax.set_yticks(np.arange(0,22,5))
            ax.tick_params(axis='x',labelrotation=45,labelsize=8,pad=1)
            ax.tick_params(axis='y',labelsize=9,pad=1)
            ax.set_title(str(round(ltimes[i])),fontsize=10)
            ax.axhline(y=10,c='black',linewidth=0.8,linestyle='--')

    return None


def sharpness(ds,ltimes):
    ltimes = np.array(ltimes,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.axhline(y=ds['width50_clim'].isel(index=0).values,c='black',linewidth=1,zorder=1)
    ax.set_xlabel('Lead time (days)')
    ax.set_ylabel('Width (m/s)')

    for l in range(len(ltimes)):
        ax.boxplot(ds['width50'].isel(leadtime=l).dropna(dim='index'),positions=[ltimes[l]],patch_artist=True,showfliers=False,medianprops=dict(color='red'))
        ax.boxplot(ds['width50_gefs'].isel(leadtime=l).dropna(dim='index'),positions=[ltimes[l]+0.18],patch_artist=True,showfliers=False,boxprops=dict(facecolor='pink'),medianprops=dict(color='red'))

    ax.set_xticks(range(6,16))
    ax.set_xticklabels(range(6,16))

    plt.show()

    print('Done')

    return None


def do_plots(ds):
    crpss = 1 - ds['crps']/ds['crps_clim']
    crpss_gefs = 1 - ds['crps_gefs']/ds['crps_clim']

    pit(ds,t_step_plot)

    box_plots(crpss,crpss_gefs,t_step_plot)
    violin_plots(xr.merge([crpss.to_dataset(name='crpss'),crpss_gefs.to_dataset(name='crpss_gefs')]),t_step_plot)


    return None



if __name__ == '__main__':
    res = 6 # time resolution in hours
    res_str = str(res)+'h'
    train_period=slice("2009-10-01","2020-09-23")
    val_period = slice("2020-10-01","2022-03-31")
    test_period_gefs = slice("2022-03-25","2024-03-25")  # 6 days earlier so that dates line up
    test_period = slice("2022-04-01","2024-04-01")
    seas = 'JJA'
    level = 850
    do = False
    agg = 1

    # setup
    Nx = 5
    Ny = 5
    N_nodes = Nx * Ny
    title = '24h-'+seas+'-'+str(level)+'-final'
    lat_offset = 11
    lon_offset = 20
    lat = np.arange(55-lat_offset,55+lat_offset+0.5,0.5)[::-1]
    lon = np.arange(240-lon_offset,240+lon_offset+0.5,0.5)

    # this is for lead time. if res > 6H, t_step is slice so that we pick up multiple z forecasts
    t_step = []
    if res <= 24:
        for d in range(6,15): # each day in week 2
            if res == 24:
                t_step.append(slice(np.array(int((d*24)*1e9*60*60),dtype='timedelta64[ns]'), 
                                    np.array(int(((d+0.9)*24)*1e9*60*60),dtype='timedelta64[ns]')))
            elif res == 12:
                t_step.append(slice(np.array(int((d*24)*1e9*60*60),dtype='timedelta64[ns]'), 
                                    np.array(int(((d+0.48)*24)*1e9*60*60),dtype='timedelta64[ns]')))
                t_step.append(slice(np.array(int(((d+0.5)*24)*1e9*60*60),dtype='timedelta64[ns]'),  
                                    np.array(int(((d+0.9)*24)*1e9*60*60),dtype='timedelta64[ns]')))
            elif res == 6:
                t_step.append(np.array(int(d*24*1e9*60*60),dtype='timedelta64[ns]'))
                t_step.append(np.array(int((d*24+6)*1e9*60*60),dtype='timedelta64[ns]'))
                t_step.append(np.array(int((d*24+12)*1e9*60*60),dtype='timedelta64[ns]'))
                t_step.append(np.array(int((d*24+18)*1e9*60*60),dtype='timedelta64[ns]'))
            else:
                print('only configured for 6h, 12h, and 24h right now')
                raise ValueError
    elif res == 48:
        t_step.append(slice(np.array(int((7*24)*1e9*60*60),dtype='timedelta64[ns]'),  # day 7,8
                                np.array(int(((8+0.9)*24)*1e9*60*60),dtype='timedelta64[ns]')))
        t_step.append(slice(np.array(int((9*24)*1e9*60*60),dtype='timedelta64[ns]'),  # day 9,10
                                np.array(int(((10+0.9)*24)*1e9*60*60),dtype='timedelta64[ns]')))
        t_step.append(slice(np.array(int((11*24)*1e9*60*60),dtype='timedelta64[ns]'),  # day 11,12
                                np.array(int(((12+0.9)*24)*1e9*60*60),dtype='timedelta64[ns]')))
        t_step.append(slice(np.array(int((13*24)*1e9*60*60),dtype='timedelta64[ns]'),  # day 13,14
                                np.array(int(((14+0.9)*24)*1e9*60*60),dtype='timedelta64[ns]')))
    else:
        print('Invalid resolution')
        raise ValueError
    
    t_step_plot = []
    if res > 6:
        for l in t_step:
            t_step_plot.append(l.start)
    else:
        t_step_plot = t_step

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
    obs_full = None  # free up memory

    # open som and dists
    print('Opening map and doing forecast...')
    with open('trained-map-'+title[4:]+'.pkl','rb') as handle:
        som = pickle.load(handle)
    indices = np.arange(N_nodes).reshape(Nx,Ny).T.flatten()

    with open('distributions-'+title[4:]+'.pkl','rb') as handle:
        distributions = pickle.load(handle)

    #plot_som(Nx, Ny, som.z_raw, indices)
    #plot_distributions(distributions)


    # era data (used for validation - finding bad nodes)
    if level == 850:
        era = xr.open_dataset('era-850-2009-2022.nc').sel(latitude=lat,longitude=lon-360)
    else:
        era = xr.open_dataset('era-2009-2022-a.nc').sel(latitude=lat,longitude=lon-360,level=level)
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

    # do forecasting, stats
    bad_nodes = era_stats(agg)

    stats_all = forecast()

    print('Getting gefs stats...')
    gefs = xr.open_dataset('~/Nextcloud/thesis/gefs-wind-all-interpolated.nc').sel(time=test_period_gefs)
    gefs = resample_mean(gefs,'gefs-wind',res) # resampled in the 'step' dimension
    stats_all = gefs_stats(gefs,stats_all)

    print('Doing clim stats')
    stats_all = clim_stats(stats_all)
    #plot_discarded(discarded,t_step_plot)

    #box_plots(crps,crps_clim,crps_gefs,t_step_plot)
    do_plots(stats_all)

    print('Done')


    
