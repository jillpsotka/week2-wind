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
import sys
import warnings
warnings.filterwarnings("ignore")
import math
from seaborn import violinplot as violin
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import statsmodels.stats.descriptivestats as dstats

# testing and plotting

mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['figure.dpi'] = 250
mpl.rcParams['image.cmap'] = 'cmc.lipari'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#0072B2","#CC79A7","#BFBFBF","#4268B3"])


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
    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0,member=0).wind,linewidth=0.4,c=colors[2])
    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0,member=1).wind,linewidth=0.4,c=colors[2])
    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0,member=2).wind,linewidth=0.4,c=colors[2])
    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0,member=3).wind,linewidth=0.4,c=colors[2])
    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0,member=4).wind,linewidth=0.4,c=colors[2])
    

    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0).wind.mean(dim='member'),linewidth=1,c='black')

    lower = gefs.sel(time=x_axis).isel(step=0).wind.min(dim='member')
    upper = gefs.sel(time=x_axis).isel(step=0).wind.max(dim='member')
    plt.fill_between(x_axis, lower, upper, color='cyan', alpha=0.2)

    plt.xticks(rotation = 45)
    plt.ylabel('Wind speed (m/s)')
    plt.show()

def plot_som_6x4(Nx1, Ny1, z, indices):
    proj=ccrs.PlateCarree()
    vmin = np.min(z)
    vmax = np.max(z)  # colorbar range
    Nx = Ny1
    Ny = Nx1
    fig, axes = plt.subplots(nrows=Ny, ncols=Nx,sharex=True,sharey='row',layout='constrained',figsize=(Nx*2,Ny*1.8),
                             subplot_kw={'projection': proj, 'aspect':1.7},gridspec_kw = {'wspace':0.01, 'hspace':0.08})
    im = comap.ScalarMappable(norm=Normalize(vmin,vmax),cmap='cmc.lipari')
    for kk, ax in enumerate(axes.flatten()):
        if lat_offset == 9:
            var = z[indices[kk],:].reshape(37,65)
            ax.set_extent(([222,258,44.75,64]))
        else:
            var = z[indices[kk],:].reshape(45,81)
            ax.set_extent(([219,261,43.25,65.25]))
        ax.set_title('Node '+str(indices[kk]+1),fontsize=15)      
        
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
        # gl.xlabel_style = {'size': 7,'rotation':40}
        # gl.ylabel_style = {'size': 7} 
        # if kk > (Ny*Nx) - Nx - 1:
        #     gl.bottom_labels = False
        # if kk % Nx == 0:
        #     gl.left_labels = False

    #cbar_ax = fig.add_axes([0.05, 0.07, 0.45*Nx, 0.03])
    cbar = fig.colorbar(im,ax=axes,fraction=0.045, shrink=0.65,pad=0.05,location='right')
    cbar.set_label('1000-hPa Geopotential Height Anomaly (m)',size=15)
    cbar.ax.tick_params(labelsize=13)

    #plt.suptitle('z500 clusters')
    plt.savefig('plots/som-'+seas+'-2x2-final.pdf')

    return None

def plot_som(Nx1, Ny1, z, indices):
    proj=ccrs.PlateCarree()
    vmin = np.min(z)
    vmax = np.max(z)  # colorbar range
    Nx = Ny1
    Ny = Nx1
    fig, axes = plt.subplots(nrows=int(Ny/2+1), ncols=int(Nx*2+1),sharex=True,sharey='row',figsize=(4,7),layout='constrained',
                             subplot_kw={'projection': proj, 'aspect':1.7},gridspec_kw = {'width_ratios':[0.5,0.5,0.2,0.5,0.5]})
    im = comap.ScalarMappable(norm=Normalize(vmin,vmax),cmap='cmc.lipari')
    for kk, ax in enumerate(axes.flatten()):
        if (kk-2) % 5 == 0:
            ax.axis('off')
            continue
        elif kk > 37:
            ax.axis('off')
            continue
        elif kk > 2:
            kk -= ((kk-3) // 5)+1
        if lat_offset == 9:
            var = z[indices[kk],:].reshape(37,65)
            ax.set_extent(([222,258,44.75,64]))
        else:
            var = z[indices[kk],:].reshape(45,81)
            ax.set_extent(([219,261,43.25,65.25]))
        ax.set_title('Node '+str(indices[kk]+1),fontsize=10)      
        
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
        # gl.xlabel_style = {'size': 7,'rotation':40}
        # gl.ylabel_style = {'size': 7} 
        # if kk > (Ny*Nx) - Nx - 1:
        #     gl.bottom_labels = False
        # if kk % Nx == 0:
        #     gl.left_labels = False

    #cbar_ax = fig.add_axes([0.05, 0.07, 0.45*Nx, 0.03])
    cbar = fig.colorbar(im,ax=axes,fraction=0.045, shrink=0.65,pad=0.05,location='right')
    cbar.set_label('1000-hPa Geopotential Height Anomaly (m)',size=10)
    cbar.ax.tick_params(labelsize=9)

    #plt.suptitle('z500 clusters')
    plt.savefig('plots/som-'+seas+'-2x2-final.pdf')

    return None

 
def plot_distributions(distributions):
        
    fig, axes = plt.subplots(nrows=Nx, ncols=Ny, figsize=(Ny*1.75,Nx*1.7),sharex=True,sharey=True,
                             layout='constrained',subplot_kw={'aspect':48},gridspec_kw = {'wspace':0.09, 'hspace':0.1})
    vmin = np.nanmin(obs_train.Wind.values)
    vmax = np.nanmax(obs_train.Wind.values)
    means = []
    
    for i, ax in enumerate(axes.flatten()):
        distribution = distributions[indices[i]]  # wind obs that belong to this node -> this is our distribution
        ax.hist(distribution, range=(vmin,vmax),color='black',
                weights=np.zeros_like(distribution) + 1. / distribution.size)
        dist_mean = np.mean(distribution)
        means.append(dist_mean)
        ax.set_title('Node '+str(indices[i]+1),fontsize=10)     
        ind = int((dist_mean-2.8)*(256)/(11-2.8)) # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        col_ar = cm.lipari.colors[ind]
        col_tuple = tuple(col_ar)
        ax.patch.set_facecolor(col_tuple)
        ax.patch.set_alpha(0.9)
        distributions.append(distribution)
        ax.tick_params(axis='both',labelsize=9)
        if dist_mean < 10:
            ax.text(3,0.36,('Mean = '+str(round(dist_mean,1))+'m/s'),fontsize=9)
        else:
            ax.text(2,0.36,('Mean = '+str(round(dist_mean,1))+'m/s'),fontsize=9)
        if i > (Nx*Ny) - Ny - 1:
            ax.set_xlabel('(m/s)',fontsize=11)
        if i % Ny == 0:
            ax.set_ylabel('Frequency',fontsize=11)

    im = comap.ScalarMappable(norm=Normalize(3,11),cmap='cmc.lipari')

    cbar = fig.colorbar(im,ax=axes,fraction=0.046, pad=0.05,shrink=0.6,location='right')
    cbar.set_label('Mean Wind Speed (m/s)',size=14)
    cbar.ax.tick_params(labelsize=11)
    #plt.show()
    plt.savefig('plots/dist-'+seas+'-2x2-final.pdf')

    return None


def plot_distributions_6x4(distributions):
        
    fig, axes = plt.subplots(nrows=Nx, ncols=Ny, figsize=(Ny*1.75,Nx*1.7),sharex=True,sharey=True,
                             layout='constrained',subplot_kw={'aspect':48},gridspec_kw = {'wspace':0.09, 'hspace':0.1})
    vmin = np.nanmin(obs_train.Wind.values)
    vmax = np.nanmax(obs_train.Wind.values)
    means = []
    
    for i, ax in enumerate(axes.flatten()):
        distribution = distributions[indices[i]]  # wind obs that belong to this node -> this is our distribution
        ax.hist(distribution, range=(vmin,vmax),color='black',
                weights=np.zeros_like(distribution) + 1. / distribution.size)
        dist_mean = np.mean(distribution)
        means.append(dist_mean)
        ax.set_title('Node '+str(indices[i]+1),fontsize=10) 
        ind = int((dist_mean-2.8)*(256)/(11-2.8)) # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        col_ar = cm.lipari.colors[ind]
        col_tuple = tuple(col_ar)
        ax.patch.set_facecolor(col_tuple)
        ax.patch.set_alpha(0.9)
        distributions.append(distribution)
        ax.tick_params(axis='both',labelsize=9)
        if dist_mean < 10:
            ax.text(3,0.36,('Mean = '+str(round(dist_mean,1))+'m/s'),fontsize=9)
        else:
            ax.text(2,0.36,('Mean = '+str(round(dist_mean,1))+'m/s'),fontsize=9)
        if i > (Nx*Ny) - Ny - 1:
            ax.set_xlabel('(m/s)',fontsize=11)
        if i % Ny == 0:
            ax.set_ylabel('Frequency',fontsize=11)

    im = comap.ScalarMappable(norm=Normalize(3,11),cmap='cmc.lipari')

    cbar = fig.colorbar(im,ax=axes,fraction=0.046, pad=0.05,shrink=0.6,location='right')
    cbar.set_label('Mean Wind Speed (m/s)',size=14)
    cbar.ax.tick_params(labelsize=11)
    #plt.show()
    plt.savefig('plots/dist-'+seas+'-2x2-final.pdf')

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
        thresh = 0
    elif aggression == 1:
        thresh = 0.1
    elif aggression == 2:
        thresh = 0.25
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
        crpss = np.nanmedian(1-crps_som[j,:]/crps_clim[j,:])
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
        bmus_mems = np.empty((len(time_axis),num_members,len(leads)))
        # bmus_mems = np.empty((int(len(obsv_loc.resample(index='24H').mean().index)),
        #                   len(files),len(leads)))
        # bmus mems is shape (days, members, lead times)
        bmus_mems.fill(np.nan)

        dist_xr = xr.DataArray(data=bmus_mems,dims=['index','member','leadtime'],
            coords=dict(index=time_axis,member=range(1,num_members+1),leadtime=leads))
        mem = 1
        for gefs_file in files[:num_members]:  # each member
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
            mem += 1
        dist_xr = dist_xr.dropna(dim='index',how='all')

        dist_xr.to_netcdf('forecast-'+title+'.nc') 

        print('Done doing forecast')
        sys.exit()
    else:
        dist_xr = xr.load_dataarray('forecast-'+title+'.nc')
    
    # get stats for each lead time
    r_list = []
    crps_list = []
    bs50_list = []
    mae_list = []
    bias_list = []
    discarded = []
    number = []

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
            # try commenting out this next if block > if still bad performance then something likely wrong in crpss calculations
            if res > 6:
                if (date-t.start) in dist_arr.index.values:
                    for c in dist_arr.sel(index=date-t.start).values.T.squeeze():  # each date in here will have a list of memberS?
                        if np.count_nonzero(np.isnan(c)) > 0.5*len(dist_xr.member):  # if many of the members are bad nodes
                            dist_list.append([np.nan])
                        else:
                            dist_list.append(np.concatenate([distributions[int(e)] for e in c if ~np.isnan(e)]))
                else:
                    obsv_loc['Wind'][kk] = np.nan
                    continue
                dis_count = 0
                for l in dist_list:
                    if np.isnan(l[0]):
                        dis_count += 1
                if dis_count > 0.75*len(dist_list):
                    dist_list = [np.array([np.nan])]

            else:
                if (date-t) in dist_arr.index.values:
                    c = dist_arr.sel(index=date-t).values
                    if np.count_nonzero(np.isnan(c)) > 0.4*len(dist_xr.member):  # if most of the members are bad nodes
                        dist_list.append([np.nan])
                    else:
                        dist_list.append(np.concatenate([distributions[int(e)] for e in c if ~np.isnan(e)]))
                else:
                    obsv_loc['Wind'][kk] = np.nan
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
        discarded.append(nan_count / (nan_count + len(dists_time)))
        number.append(nan_count + len(dists_time))
        
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
            # w95=np.percentile(dists_nonan, 95)
            # w5=np.percentile(dists_nonan, 5)
            # w75=np.percentile(dists_nonan, 75)
            # w25=np.percentile(dists_nonan, 25)
            # stats_xr['width90'].loc[dict(index=date,leadtime=t)]= w95-w5
            # stats_xr['width50'].loc[dict(index=date,leadtime=t)]= w75-w25

        # correlation between distribution means and obs through time
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(dist_means, obsv_cut.Wind)
        except ValueError:  # throws this error if only 1 node is 'good'
            r_value = np.nan
            p_value = 1
        if p_value > 0.05:
            bad_r_som.append(tt)
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
            s, pvalue = dstats.sign_test((prob-target)**2,0.25)
            brier = brier_score_loss(target, prob)
            if pvalue > 0.05:
                bad_brier_som.append(tt)
        bs50 = brier
        # add to lists
        r_list.append(r_value)
        crps_list.append(crps_som)
        bs50_list.append(bs50)
        bias_list.append(bias)
        mae_list.append(mae)

    stats_xr['discarded'] = (['leadtime'],discarded)
    stats_xr['number'] = (['leadtime'],number)
    stats_xr['r'] = (['leadtime'],r_list)
    stats_xr['mae'] = (['leadtime'],mae_list)
    stats_xr['bs50'] = (['leadtime'],bs50_list)
    return stats_xr


def clim_dists(ds):
            
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4,4),sharex=True,sharey=True)
    vmin = np.nanmin(obs_train.Wind.values)
    vmax = np.nanmax(obs_train.Wind.values)
    seasons = ['DJF','MAM','JJA','SON']
    means = []
    
    for i, ax in enumerate(axes.flatten()):
        distribution = ds.sel(index=ds.index.dt.season==seasons[i])  # wind obs that belong to this node -> this is our distribution
        ax.hist(distribution, range=(vmin,vmax),color='black',
                weights=np.zeros_like(distribution) + 1. / distribution.size)
        dist_mean = np.mean(distribution)
        means.append(dist_mean)
        ax.set_title(seasons[i],fontsize=7,pad=0.1)
        ind = int((dist_mean-2.8)*(256)/(12-2.8)) # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        col_ar = cm.lipari.colors[ind]
        col_tuple = tuple(col_ar)
        ax.patch.set_facecolor(col_tuple)
        ax.patch.set_alpha(0.9)
        distributions.append(distribution)
        ax.tick_params(axis='both',labelsize=6.5,pad=0.2,width=0.5,length=2)
        ax.set_yticks(np.arange(0,0.4,0.1))
        ax.set_xticks([0,5,10,15])
        # if dist_mean < 10:
        #     ax.text(0.1,0.33,('Mean='+str(round(dist_mean,1))+'m/s'),fontsize=5)
        # else:
        #     ax.text(0.1,0.33,('Mean='+str(round(dist_mean,1))+'m/s'),fontsize=5)
        if i > 23:
            ax.set_xlabel('(m/s)',fontsize=7,labelpad=0.2)
        if i % 4 == 0:
            ax.set_ylabel('Frequency',fontsize=7)

    im = comap.ScalarMappable(norm=Normalize(3,11),cmap='cmc.lipari')

    cbar = fig.colorbar(im,ax=axes,fraction=0.046, pad=0.05,shrink=0.6,location='right')
    cbar.set_label('Mean Wind Speed (m/s)',size=8)
    cbar.ax.tick_params(labelsize=7)
    #plt.show()
    plt.savefig('plots/dist-clim-2x2-final.pdf',bbox_inches='tight')

    return None


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
        ds['prob_clim'].loc[dict(index=gh)] = sum(v < ob for v in dists_nonan) / len(dists_nonan)

        #ds['width_clim'].loc[dict(index=gh)]= np.max(dists_nonan) - np.min(dists_nonan)
        #w95=np.percentile(dists_nonan, 95)
        #w5=np.percentile(dists_nonan, 5)
        # w75=np.percentile(dists_nonan, 75)
        # w25=np.percentile(dists_nonan, 25)
        # ds['width50_clim'].loc[dict(index=gh)]= w75-w25

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
    number = []

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

                #st['width_gefs'].loc[dict(index=obs_date,leadtime=t)]= np.max(dists_nonan) - np.min(dists_nonan)
                # w95=np.percentile(dists_nonan, 95)
                # w5=np.percentile(dists_nonan, 5)
                # w75=np.percentile(dists_nonan, 75)
                # w25=np.percentile(dists_nonan, 25)
                # st['width50_gefs'].loc[dict(index=obs_date,leadtime=t)]= w75-w25
        mae_list.append(np.mean(mae))
        bias_list.append(np.mean(bias))
        crps_list.append(np.array(crps))
        number.append(len(mae))

        slope, intercept, r_gefs, p_gefs, std_err = stats.linregress(g_time_series, ob_time_series)
        if p_gefs> 0.05:
            bad_r_gefs.append(tt)
        r_list.append(r_gefs)
        
        splits = [50]
        bss = []
        for percentile in splits:
            split = np.percentile(obs_train.Wind.dropna(dim='index'), percentile)
            target = np.array(ob_time_series > split)  # boolean array of obs
            prob = [np.sum(d>split)/len(d[~np.isnan(d)]) for d in g_time_series_members]  # probabilities of being above threshold in each distribution
            s, pvalue = dstats.sign_test((prob-target)**2,0.25)
            brier = brier_score_loss(target, prob)
            if pvalue > 0.05:
                bad_brier_gefs.append(tt)

        bs50_list.append(brier)

    st['r_gefs'] = (['leadtime'],r_list)
    st['mae_gefs'] = (['leadtime'],mae_list)
    st['bs50_gefs'] = (['leadtime'],bs50_list)
    st['number_gefs'] = (['leadtime'],number)

    return st
            


def plot_error(som_error, gefs_error, ltimes,clim_error=False):
    ltimes = np.array(ltimes,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)

    fig, ax = plt.subplots(figsize=(7,4))

    ax.plot(ltimes, som_error, label='SOM',zorder=3,marker='o',markersize=3)
    ax.plot(ltimes, gefs_error, label='GEFS 80m wind',zorder=2,marker='o',markersize=3)
    if clim_error:
        ax.axhline(np.nanmean(clim_error), label='Climatology',c='black',zorder=1)
    ax.set_xlabel('Lead time (days)',fontsize=14)
    ax.set_ylabel('R',fontsize=14)
    ax.set_xticks(range(6,16))
    ax.xaxis.grid(True,which='both',linestyle='--')
    ax.legend(fontsize=12)
    ax.tick_params(axis='both',labelsize=12,pad=1)
    plt.tight_layout()
    plt.savefig('plots/R-'+seas+'-'+str(res)+'h.pdf',bbox_inches='tight')

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


def box_plots(som_error,gefs_error, ltimes,bad_som=[],bad_gefs=[],cl=0):
    ltimes = np.array(ltimes,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.axhline(y=cl,c='black',linewidth=1,zorder=1)
    ax.set_xlabel('Lead time (days)',fontsize=14)
    ax.tick_params(axis='both',labelsize=12,pad=1)

    blue_patch = mpatches.Patch(facecolor=colors[0], label='SOM',edgecolor='black')
    pink_patch = mpatches.Patch(facecolor=colors[1], label='GEFS 80-m wind',edgecolor='black')
    black_line = mlines.Line2D([], [], color='black',label='Climatology')

    if res == 48:
        w = 0.8
        a=0.5
        b= 1.35
    else:
        w = 0.35
        a=0.2
        b=0.6

    for l in range(len(som_error.leadtime)):
        if l in bad_som:
            ax.boxplot(som_error.isel(leadtime=l).dropna(dim='index'),positions=[ltimes[l]+a],widths=w,patch_artist=True,showfliers=False,boxprops=dict(facecolor=colors[2]),medianprops=dict(color='black'))
        else:
            ax.boxplot(som_error.isel(leadtime=l).dropna(dim='index'),positions=[ltimes[l]+a],widths=w,patch_artist=True,showfliers=False,medianprops=dict(color='black'))
        if l in bad_gefs:
            ax.boxplot(gefs_error.isel(leadtime=l).dropna(dim='index'),positions=[ltimes[l]+b],widths=w,patch_artist=True,showfliers=False,boxprops=dict(facecolor=colors[2]),medianprops=dict(color='black'))
        else:
            ax.boxplot(gefs_error.isel(leadtime=l).dropna(dim='index'),positions=[ltimes[l]+b],widths=w,patch_artist=True,showfliers=False,boxprops=dict(facecolor=colors[1]),medianprops=dict(color='black'))

    ax.set_xticks(range(6,16))
    ax.set_xticklabels(range(6,16))
    if cl == 0:
        ax.set_ylabel('CRPSS',fontsize=14)
        ax.legend(handles=[blue_patch, pink_patch],loc='lower right')
        plt.savefig('plots/box-'+seas+'-'+str(res)+'h.pdf',bbox_inches='tight')
    else:
        ax.set_ylabel('Width (m/s)',fontsize=14)
        ax.legend(handles=[blue_patch, pink_patch,black_line],loc='upper right',borderaxespad=0.1)
        plt.savefig('plots/sharp-'+seas+'-'+str(res)+'h.pdf',bbox_inches='tight')

    print('Done')


def box_plots_det(som_error,gefs_error, clim_error,ltimes,tit='ae'):
    ltimes = np.array(ltimes,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlabel('Lead time (days)',fontsize=14)
    ax.set_ylabel('Absolute error (m/s)',fontsize=14)
    ax.tick_params(axis='both',labelsize=12,pad=1)

    blue_patch = mpatches.Patch(facecolor="#0072B2", label='SOM',edgecolor='black')
    pink_patch = mpatches.Patch(facecolor=colors[1], label='GEFS 80-m wind',edgecolor='black')
    black_patch = mpatches.Patch(facecolor='black', label='Climatology',edgecolor='black')

    if res == 48:
        w = 0.5
        a=0.5
        b= 1.35
    else:
        w = 0.2
        a=0.25
        b=0.5
        c=0.75

    for l in range(len(som_error.leadtime)):
        ax.boxplot(som_error.isel(leadtime=l).dropna(dim='index'),positions=[ltimes[l]+a],widths=w,patch_artist=True,showfliers=False,medianprops=dict(color='black'))
        ax.boxplot(gefs_error.isel(leadtime=l).dropna(dim='index'),positions=[ltimes[l]+b],widths=w,patch_artist=True,showfliers=False,boxprops=dict(facecolor=colors[1]),medianprops=dict(color='black'))
        ax.boxplot(clim_error,positions=[ltimes[l]+c],widths=w,patch_artist=True,showfliers=False,boxprops=dict(facecolor='black'),medianprops=dict(color='black'))

    ax.set_xticks(range(6,16))
    ax.set_xticklabels(range(6,16))

    plt.legend(handles=[blue_patch, pink_patch,black_patch],loc='upper right')

    plt.savefig('plots/box-'+tit+'-'+seas+'-'+str(res)+'h.pdf',bbox_inches='tight')

    print('Done')



def box_plots_6(som_error, gefs_error, ltimes,bad_som=[],bad_gefs=[],cl=0):
    ltimes = np.array(ltimes,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)
    ml = mticker.MultipleLocator(0.5)
    ax_tits = ['00Z','06Z','12Z','18Z']

    fig, axes = plt.subplots(4,1,figsize=(5,8),sharex=True,gridspec_kw = {'hspace':0.23})
    blue_patch = mpatches.Patch(facecolor=colors[0], label='SOM',edgecolor='black')
    pink_patch = mpatches.Patch(facecolor=colors[1], label='GEFS 80-m wind',edgecolor='black')
    black_line = mlines.Line2D([], [], color='black',label='Climatology')

    for i , ax in enumerate(axes):
        ax.axhline(y=cl,c='black',linewidth=1,zorder=1)
        if i == 3:
            ax.set_xlabel('Lead time (days)',fontsize=11,labelpad=1.5)

        for l in range(int(len(som_error.leadtime)/4)):
            if l*4+i in bad_som:
                ax.boxplot(som_error.isel(leadtime=(l*4+i)).dropna(dim='index'),positions=[ltimes[l*4+i]-0.1],patch_artist=True,showfliers=False,boxprops=dict(facecolor=colors[2]),medianprops=dict(color='black'))
            else:
                ax.boxplot(som_error.isel(leadtime=(l*4+i)).dropna(dim='index'),positions=[ltimes[l*4+i]-0.1],patch_artist=True,showfliers=False,medianprops=dict(color='black'))

            if l*4+i in bad_gefs:
                ax.boxplot(gefs_error.isel(leadtime=(l*4+i)).dropna(dim='index'),positions=[ltimes[l*4+i]+0.1],patch_artist=True,showfliers=False,boxprops=dict(facecolor=colors[2]),medianprops=dict(color='black'))
            else:
                ax.boxplot(gefs_error.isel(leadtime=(l*4+i)).dropna(dim='index'),positions=[ltimes[l*4+i]+0.1],patch_artist=True,showfliers=False,boxprops=dict(facecolor=colors[1]),medianprops=dict(color='black'))

        ax.set_xticks(range(6,16))
        ax.set_xticklabels(range(6,16))
        if cl == 0:
            ax.set_yticks(np.arange(-1,1.1,1))
            ax.set_yticklabels(np.arange(-1,1.1,1))
            ax.set_ylim(bottom=-2)
            ax.yaxis.set_minor_locator(ml)
            ax.set_ylabel('CRPSS',fontsize=11)
        else:
            ax.set_ylabel('Width (m/s)',fontsize=11)
        ax.tick_params(axis='both',labelsize=10,pad=1)
        ax.set_title(ax_tits[i],fontsize=12,pad=1.3)

    if cl == 0:
        fig.legend(handles=[blue_patch, pink_patch],loc='upper right',borderaxespad=1.5)
        plt.savefig('plots/box-'+seas+'-'+str(res)+'h.pdf',bbox_inches='tight')
    else:
        fig.legend(handles=[blue_patch, pink_patch,black_line],loc='upper right',borderaxespad=0.7)
        plt.savefig('plots/sharp-'+seas+'-'+str(res)+'h.pdf',bbox_inches='tight')

    print('Done')



def box_plots_12(som_error, gefs_error, ltimes,bad_som=[],bad_gefs=[],cl=0):
    ltimes = np.array(ltimes,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)
    ml = mticker.MultipleLocator(0.5)
    ax_tits = ['00Z','12Z']

    fig, axes = plt.subplots(2,1,figsize=(5,6),sharex=True,gridspec_kw = {'hspace':0.15})
    blue_patch = mpatches.Patch(facecolor=colors[0], label='SOM',edgecolor='black')
    pink_patch = mpatches.Patch(facecolor=colors[1], label='GEFS 80-m wind',edgecolor='black')
    black_line = mlines.Line2D([], [], color='black',label='Climatology')

    for i , ax in enumerate(axes):
        ax.axhline(y=cl,c='black',linewidth=1,zorder=1)
        if i == 1:
            ax.set_xlabel('Lead time (days)',fontsize=11)
        ax.set_ylabel('CRPSS',fontsize=11)

        for l in range(int(len(som_error.leadtime)/2)):
            if l*2+i in bad_som:
                ax.boxplot(som_error.isel(leadtime=(l*2+i)).dropna(dim='index'),widths=0.25,positions=[ltimes[l*2+i]-0.15],patch_artist=True,showfliers=False,boxprops=dict(facecolor=colors[2]),medianprops=dict(color='black'))
            else:
                ax.boxplot(som_error.isel(leadtime=(l*2+i)).dropna(dim='index'),widths=0.25,positions=[ltimes[l*2+i]-0.15],patch_artist=True,showfliers=False,boxprops=dict(facecolor=colors[0]),medianprops=dict(color='black'))
            if l*2+i in bad_gefs:
                ax.boxplot(gefs_error.isel(leadtime=(l*2+i)).dropna(dim='index'),widths=0.25,positions=[ltimes[l*2+i]+0.15],patch_artist=True,showfliers=False,boxprops=dict(facecolor=colors[2]),medianprops=dict(color='black'))
            else:
                ax.boxplot(gefs_error.isel(leadtime=(l*2+i)).dropna(dim='index'),widths=0.25,positions=[ltimes[l*2+i]+0.15],patch_artist=True,showfliers=False,boxprops=dict(facecolor=colors[1]),medianprops=dict(color='black'))

        ax.set_xticks(range(6,16))
        ax.set_xticklabels(range(6,16))
        if cl == 0:
            ax.set_yticks(np.arange(-1,1.1,1))
            ax.set_yticklabels(np.arange(-1,1.1,1))
            ax.set_ylim(bottom=-2)
            ax.yaxis.set_minor_locator(ml)
            ax.set_ylabel('CRPSS',fontsize=11)
        else:
            ax.set_ylabel('Width (m/s)',fontsize=11)
        ax.set_title(ax_tits[i],fontsize=12)

    if cl == 0:
        fig.legend(handles=[blue_patch, pink_patch],loc='upper right',borderaxespad=0.1)
        plt.savefig('plots/box-'+seas+'-'+str(res)+'h.pdf',bbox_inches='tight')
    else:
        fig.legend(handles=[blue_patch, pink_patch,black_line],loc='upper right',borderaxespad=0.01)
        plt.savefig('plots/sharp-'+seas+'-'+str(res)+'h.pdf',bbox_inches='tight')

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


def pit(ds,ltimes1,gefs_opt=False,tit=''):
    ltimes = np.array(ltimes1,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)

    fig, axes = plt.subplots(nrows=3, ncols=math.ceil(len(ltimes)/3), figsize=(7,5),sharey=True,sharex=True,
                             gridspec_kw = {'wspace':0.13, 'hspace':0.31})
    
    for i, ax in enumerate(axes.flatten()):
        if i < len(ltimes):
            if gefs_opt:
                probs = ds['prob_gefs'].sel(leadtime=ltimes1[i])[~np.isnan(ds['prob_gefs'].sel(leadtime=ltimes1[i]))]
                ax.hist(probs,color=colors[1],bins=np.arange(0,1.1,0.1),weights=(np.zeros_like(probs) + 1./probs.size)*100,edgecolor='black')
            else:
                probs = ds['prob'].sel(leadtime=ltimes1[i])[~np.isnan(ds['prob'].sel(leadtime=ltimes1[i]))]
                ax.hist(probs,color=colors[0],bins=np.arange(0,1.1,0.1),weights=(np.zeros_like(probs) + 1./probs.size)*100,edgecolor='black')
            ax.set_xticks(np.arange(0,1.1,0.1))
            ax.set_xticklabels(np.arange(0,101,10))
            ax.set_yticks(np.arange(0,32,5))
            ax.tick_params(axis='x',labelrotation=60,labelsize=10,pad=1)
            ax.tick_params(axis='y',labelsize=10,pad=1)
            ax.set_title('Day '+str(round(ltimes[i])),fontsize=12,pad=1.3)
            ax.axhline(y=10,c='black',linewidth=0.8,linestyle='--')
            if i%3 == 0:
                ax.set_ylabel('$\%$ of forecasts',fontsize=12)
            if i > 5:
                ax.set_xlabel('Percentile bin',fontsize=12)
    if tit:
        plt.suptitle(tit)
    if gefs_opt:
        plt.savefig('plots/pit-'+seas+'-GEFS-'+str(res)+'h'+tit+'.pdf',bbox_inches='tight')
    else:
        plt.savefig('plots/pit-'+seas+'-'+str(res)+'h'+tit+'.pdf',bbox_inches='tight')

    return None

def pit_clim(ds):
    if res < 24:
        if res == 6:
            fig, axes = plt.subplots(nrows=2, ncols=2,layout='constrained')
            tit = ['00Z','06Z','12Z','18Z']
        elif res == 12:
            tit = ['00Z','12Z']
            fig, axes = plt.subplots(nrows=1, ncols=2,layout='constrained')

        for i, ax in enumerate(axes.flatten()):
            h = int(tit[i][:2])
            probs = ds['prob_clim'].sel(index=pd.to_datetime(ds.index).hour==h).dropna(dim='index')
                        
            ax.hist(probs,color='dimgrey',bins=np.arange(0,1.1,0.1),weights=(np.zeros_like(probs) + 1./probs.size)*100,edgecolor='black')
            ax.set_xticks(np.arange(0,1.1,0.1))
            ax.set_xticklabels(np.arange(0,101,10))
            ax.set_yticks(np.arange(0,32,5))
            ax.tick_params(axis='x',labelrotation=60,labelsize=14,pad=0)
            ax.tick_params(axis='y',labelsize=14,pad=1)
            ax.axhline(y=10,c='black',linewidth=0.8,linestyle='--')
            ax.set_ylabel('$\%$ of forecasts',fontsize=16)
            ax.set_xlabel('Percentile bin',fontsize=15)
            ax.set_title(tit[i])
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    
        probs = ds['prob_clim'][~np.isnan(ds['prob_clim'])]
                    
        ax.hist(probs,color='dimgrey',bins=np.arange(0,1.1,0.1),weights=(np.zeros_like(probs) + 1./probs.size)*100,edgecolor='black')
        ax.set_xticks(np.arange(0,1.1,0.1))
        ax.set_xticklabels(np.arange(0,101,10))
        ax.set_yticks(np.arange(0,32,5))
        ax.tick_params(axis='x',labelrotation=60,labelsize=14,pad=0)
        ax.tick_params(axis='y',labelsize=14,pad=1)
        ax.axhline(y=10,c='black',linewidth=0.8,linestyle='--')
        ax.set_ylabel('$\%$ of forecasts',fontsize=16)
        ax.set_xlabel('Percentile bin',fontsize=16)

    plt.savefig('plots/pit-'+seas+'-clim-'+str(res)+'h.pdf',bbox_inches='tight')

def pit_summary(ds):
    probs = [ds['prob'].isel(leadtime=0).dropna(dim='index'),ds['prob_gefs'].isel(leadtime=0).dropna(dim='index'),
             ds['prob_clim'].dropna(dim='index')]
    fig, axes = plt.subplots(2,2,figsize=(5,4),layout='constrained')
    tits = ['SOM','GEFS','Climatology']
    for i, ax in enumerate(axes.flatten()):
        if i == 3:
            ax.axis('off')
            continue
        ax.hist(probs[i],color=colors[i],bins=np.arange(0,1.1,0.1),weights=(np.zeros_like(probs[i]) + 1./probs[i].size)*100,edgecolor='black')
        ax.set_xticks(np.arange(0,1.1,0.1))
        ax.set_xticklabels(np.arange(0,101,10))
        ax.set_yticks(np.arange(0,32,5))
        ax.tick_params(axis='x',labelrotation=60,labelsize=9,pad=0)
        ax.tick_params(axis='y',labelsize=9,pad=1)
        ax.axhline(y=10,c='black',linewidth=0.8,linestyle='--')
        ax.set_ylabel('$\%$ of forecasts',fontsize=11)
        if i != 0:
            ax.set_xlabel('Percentile bin',fontsize=11)
        ax.set_title(tits[i],pad=1)
        plt.savefig('plots/pit-'+seas+'-summary-'+str(res)+'h.pdf',bbox_inches='tight')

def do_plots(ds):
    crpss = 1 - ds['crps']/ds['crps_clim']
    crpss_gefs = 1 - ds['crps_gefs']/ds['crps_clim']

    bss50 = 1 - ds['bs50'] / 0.25
    bss50_gefs = 1 - ds['bs50_gefs'] / 0.25

    bad_som_crpss = []
    bad_gefs_crpss = []
    bad_som_mae = []
    bad_gefs_mae = []
    bad_som_bias = []
    bad_gefs_bias = []
    bad_sharp_som = []
    bad_sharp_gefs = []
    for i in range(len(ds.leadtime)):
        l = ds.leadtime[i]
        h = pd.to_datetime(l.astype('datetime64[ns]').values).hour
        #crpss
        s,p = dstats.sign_test(crpss.sel(leadtime=l).dropna(dim='index'))
        if p > 0.05:
            bad_som_crpss.append(i)
        s,p = dstats.sign_test(crpss_gefs.sel(leadtime=l).dropna(dim='index'))
        if p > 0.05:
            bad_gefs_crpss.append(i)

        # mae
        s,p = stats.mannwhitneyu(abs(ds['bias'].sel(leadtime=l).dropna(dim='index')),abs(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h).dropna(dim='index')))
        if p > 0.05:
            bad_som_mae.append(i)
        s,p = stats.mannwhitneyu(abs(ds['bias_gefs'].sel(leadtime=l).dropna(dim='index')),abs(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h).dropna(dim='index')))
        if p > 0.05:
            bad_gefs_mae.append(i)

        # bias
        s,p = stats.mannwhitneyu(ds['bias'].sel(leadtime=l).dropna(dim='index'),ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h).dropna(dim='index'))
        if p > 0.05:
            bad_som_bias.append(i)
        s,p = stats.mannwhitneyu(ds['bias_gefs'].sel(leadtime=l).dropna(dim='index'),ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h).dropna(dim='index'))
        if p > 0.05:
            bad_gefs_bias.append(i)

        # sharpness
        # s,p = dstats.sign_test(ds['width50'].sel(leadtime=l).dropna(dim='index'),ds['width50_clim'].isel(index=i%4).values)
        # if p > 0.05:
        #     bad_sharp_som.append(i)
        # s,p = dstats.sign_test(ds['width50_gefs'].sel(leadtime=l).dropna(dim='index'),ds['width50_clim'].isel(index=i%4).values)
        # if p > 0.05:
        #     bad_sharp_gefs.append(i)       

    # if res == 12:
    #     pit(ds,t_step_plot[::2],tit='00Z')
    #     pit(ds,t_step_plot[1::2],tit='12Z')
    #     pit(ds,t_step_plot[::2],True,tit='00Z')  # gefs pit
    #     pit(ds,t_step_plot[1::2],True,tit='12Z')
    # elif res == 6:
    #     pit(ds,t_step_plot[::4],tit='00Z')
    #     pit(ds,t_step_plot[1::4],tit='06Z')
    #     pit(ds,t_step_plot[2::4],tit='12Z')
    #     pit(ds,t_step_plot[3::4],tit='18Z')
    #     pit(ds,t_step_plot[::4],True,tit='00Z')  # gefs pit
    #     pit(ds,t_step_plot[1::4],True,tit='06Z')
    #     pit(ds,t_step_plot[2::4],True,tit='12Z')  # gefs pit
    #     pit(ds,t_step_plot[3::4],True,tit='18Z')
    # elif res >= 24:
    #     pit(ds,t_step_plot)
    #     pit(ds,t_step_plot,True)
    # pit_clim(ds)
    #pit_summary(ds)

    if res == 6:
        box_plots_6(crpss,crpss_gefs,t_step_plot,bad_som_crpss,bad_gefs_crpss)
    #     box_plots_6(ds['width50'],ds['width50_gefs'],t_step_plot,bad_sharp_som,bad_sharp_gefs,cl=ds['width50_clim'].isel(index=0).values)  # sharpness
    # elif res == 12:
    #     box_plots_12(crpss,crpss_gefs,t_step_plot,bad_som_crpss,bad_gefs_crpss)
    #     box_plots_12(ds['width50'],ds['width50_gefs'],t_step_plot,bad_sharp_som,bad_sharp_gefs,cl=ds['width50_clim'].isel(index=0).values)  # sharpness
    # else:
    #     box_plots(crpss,crpss_gefs,t_step_plot,bad_som_crpss,bad_gefs_crpss)  # crpss
    #     box_plots(ds['width50'],ds['width50_gefs'],t_step_plot,bad_sharp_som,bad_sharp_gefs,cl=ds['width50_clim'].isel(index=0).values)  # sharpness
    
    full_st = ''
    full_st_gefs = ''
    for i in range(len(ds.leadtime)):
        l = ds.leadtime[i]
        st = ''
        st_gefs = ''

        if res == 12 and i%2 == 0:
            h=0
            st+= r"\rowcolor{gray!25}\cellcolor{white}\multirow{2}{*}{"+str(6+i//2)+'} & 00&'
            st_gefs+= r"\rowcolor{gray!25}\cellcolor{white}\multirow{2}{*}{"+str(6+i//2)+'} & 00&'
        elif res == 12 and i%2 != 0:
            h=12
            st+= '& 12&'
            st_gefs+= '& 12&'
        elif res == 6 and i%4 == 0:
            h=0
            st+= r"\rowcolor{gray!25}\cellcolor{white}\multirow{4}{*}{} & 00&"
            st_gefs+= r"\rowcolor{gray!25}\cellcolor{white}\multirow{4}{*}{} & 00&"
        elif res == 6 and (i-1)%4 == 0:
            h=6
            st+= '& 06&'
            st_gefs+= '& 06&'
        elif res == 6 and (i-2)%4 == 0:
            h=12
            st+= r'\rowcolor{gray!25}\cellcolor{white}'+str(6+i//4)+'& 12&'
            st_gefs+= r'\rowcolor{gray!25}\cellcolor{white}'+str(6+i//4)+'& 12&'
        elif res == 6 and (i-3)%4 == 0:
            h=18
            st+= '& 18&'
            st_gefs+= '& 18&'
        elif res == 24:
            h=0
            st+= str(i+6) + ' & '
            st_gefs+= str(i+6) + ' & '
        elif res == 48:
            h=0
            st+= str(i*2+7)+'-'+str(i*2+8)+ ' & '
            st_gefs+= str(i*2+7)+'-'+str(i*2+8)+ ' & '
        else:
            st += '& '
            st_gefs += '& '

        # bias
        b = round(np.nanmean(ds['bias'].sel(leadtime=l)),2)
        b_gefs = round(np.nanmean(ds['bias_gefs'].sel(leadtime=l)),2)
        if i in bad_som_bias:
            st += str(b) + ' &'
        elif abs(b) < abs(round(np.nanmean(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h)),2)):
            st += '\\cellcolor{green!25}\\textbf{' + str(b) + '} &'
        else:
            st += '\\textbf{' + str(b) + '} &'

        if i in bad_gefs_bias:
            st_gefs += str(b_gefs) + ' &'
        elif abs(b_gefs) < abs(round(np.nanmean(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h)),2)):
            st_gefs += '\\cellcolor{green!25}\\textbf{' + str(b_gefs) + '} &'
        else:
            st_gefs += '\\textbf{' + str(b_gefs) + '} &'

        # mae
        b = round(np.nanmean(abs(ds['bias'].sel(leadtime=l))),2)
        b_gefs = round(np.nanmean(abs(ds['bias_gefs'].sel(leadtime=l))),2)
        if i in bad_som_mae:
            st += str(b) + ' &'
        elif b < round(np.nanmean(abs(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h))),2):
            st += '\cellcolor{green!25}\\textbf{' + str(b) + '} &'
        else:
            st += '\\textbf{' + str(b) + '} &'
        
        if i in bad_gefs_mae:
            st_gefs += str(b_gefs) + ' &'
        elif b_gefs < round(np.nanmean(abs(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h))),2):
            st_gefs += '\cellcolor{green!25}\\textbf{' + str(b_gefs) + '} &'
        else:
            st_gefs += '\\textbf{' + str(b_gefs) + '} &'

        # r
        b = round(np.nanmean(ds['r'].sel(leadtime=l)),2)
        b_gefs = round(np.nanmean(ds['r_gefs'].sel(leadtime=l)),2)
        if i in bad_r_som:
            st += str(b) + ' &'
        elif b > 0:
            st += '\cellcolor{green!25}\\textbf{' + str(b) + '} &'
        else:
            st += '\\textbf{' + str(b) + '} &'

        if i in bad_r_gefs:
            st_gefs += str(b_gefs) + ' &'
        elif b_gefs > 0:
            st_gefs += '\cellcolor{green!25}\\textbf{' + str(b_gefs) + '} &'
        else:
            st_gefs += '\\textbf{' + str(b_gefs) + '} &'

        # bs50
        b = round(np.nanmean(bss50.sel(leadtime=l)),2)
        b_gefs = round(np.nanmean(bss50_gefs.sel(leadtime=l)),2)
        if i in bad_brier_som:
            st += str(b) + ' &'
        elif b > 0:
            st += '\cellcolor{green!25}\\textbf{' + str(b) + '} &'
        else:
            st += '\\textbf{' + str(b) + '} &'

        if i in bad_brier_gefs:
            st_gefs += str(b_gefs) + ' &'
        elif b_gefs > 0:
            st_gefs += '\cellcolor{green!25}\\textbf{' + str(b_gefs) + '} &'
        else:
            st_gefs += '\\textbf{' + str(b_gefs) + '} &'

        # crpss
        b = round(np.nanmedian(crpss.sel(leadtime=l)),2)
        b_gefs = round(np.nanmedian(crpss_gefs.sel(leadtime=l)),2)
        if i in bad_som_crpss:
            st += str(b) + ' &'
        elif b > 0:
            st += '\cellcolor{green!25}\\textbf{' + str(b) + '} &'
        else:
            st += '\\textbf{' + str(b) + '} &'

        if i in bad_gefs_crpss:
            st_gefs += str(b_gefs) 
        elif b_gefs > 0:
            st_gefs += '\cellcolor{green!25}\\textbf{' + str(b_gefs) + '} &'
        else:
            st_gefs += '\\textbf{' + str(b_gefs) + '} '

        st += str(int(100-100*ds['discarded'].sel(leadtime=l).values))+r'\% \\'
        st_gefs += r'\\'
        if res == 12 and i%2 == 0:
            st += r'\cline{2-8}'
            st_gefs += r'\cline{2-7}'
        elif res == 12 and i%2 != 0:
            st += '\n'+ r'\hline'
            st_gefs += '\n' + r'\hline' 

        elif res == 6 and (i%4 == 0 or (i-1)%4==0 or (i-2)%4==0):
            st += r' \cline{2-8}'
            st_gefs += r' \cline{2-7}' 
        elif res == 6 and (i-3)%4 == 0:
            st += '\n'+ r'\hline'
            st_gefs += '\n' + r'\hline' 
        
        else:
            st += '\n' + r'\hline'
            st_gefs += '\n' + r'\hline'

        full_st+= (st+'\n')
        full_st_gefs+= (st_gefs+'\n')
    full_st = full_st[:-8]
    full_st += '\n'+r'\Xhline{2.5\arrayrulewidth}' + '\n'
    full_st_gefs = full_st_gefs[:-8]
    full_st_gefs += '\n'+r'\Xhline{2.5\arrayrulewidth}' + '\n'

    if res >= 24:
        full_st += 'Clim &'+(str(round(np.nanmean(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==0)),2))
                    + ' & ' + str(round(np.nanmean(abs(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==0))),2))
                    + r' & N/A & 0 & 0 & N/A\\'+'\n' + r'\hline')
        full_st_gefs += 'Clim &'+(str(round(np.nanmean(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==0)),2))
                    + ' & ' + str(round(np.nanmean(abs(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==0))),2))
                    + r' & N/A & 0 & 0 N/A \\'+'\n' + r'\hline')
    else:
        full_st +='Clim '
        full_st_gefs += 'Clim '
        if res == 12:
            h_list = [0,12]
        elif res == 6:
            h_list = [0,6,12,18]
        for h in h_list:
            full_st += ' &'+str(h)+' &' + (str(round(np.nanmean(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h)),2))
                        + ' & ' + str(round(np.nanmean(abs(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h))),2))
                        + r' & N/A & 0 & 0 & N/A\\'+'\n' +  r' \cline{2-8}')
            full_st_gefs += ' &'+str(h)+' &' + (str(round(np.nanmean(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h)),2))
                        + ' & ' + str(round(np.nanmean(abs(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h))),2))
                        + r' & N/A & 0 & 0 N/A \\'+'\n' +  r' \cline{2-7}')
    print('SOM \n' + full_st)
    print('GEFS \n' + full_st_gefs)
    full_st += '\n'+'\n'+str(ds['number'].isel(leadtime=0).values) + ' total forecast-obs pairs'
    full_st_gefs += '\n'+'\n'+str(ds['number_gefs'].isel(leadtime=0).values) + ' total forecast-obs pairs'

    with open(('table-'+seas+'-'+str(res)+'h.txt'), 'w') as file:
        file.write(full_st)
    with open(('table-'+seas+'-'+str(res)+'h-GEFS.txt'), 'w') as file:
        file.write(full_st_gefs)

    #violin_plots(xr.merge([crpss.to_dataset(name='crpss'),crpss_gefs.to_dataset(name='crpss_gefs')]),t_step_plot)

    return None



if __name__ == '__main__':
    #res = int(sys.argv[2]) # time resolution in hours
    res = 6
    res_str = str(res)+'h'
    train_period=slice("2009-10-01","2020-09-23")
    val_period = slice("2020-10-01","2022-03-31")
    test_period_gefs = slice("2022-03-25","2024-03-25")  # 6 days earlier so that dates line up
    test_period = slice("2022-04-01","2024-04-01")
    #test_period_gefs = slice("2020-09-24","2022-03-24")
    #test_period = slice("2020-10-01","2022-03-31")
    #seas = sys.argv[1]
    seas= 'DJF'
    if seas == "DJF" or seas == "JJA":
        level = 1000
        agg = 2
        Nx = 15
        Ny = 2
    elif seas == "SON":
        level = 850
        agg = 1
        Nx = 14
        Ny = 2
    elif seas == "MAM":
        level = 850
        agg = 2
        Nx = 6
        Ny = 4
    lat_offset = 9
    lon_offset = 16

    do = False
    num_members = 15

    # setup
    N_nodes = Nx * Ny
    title = '24h-'+seas+'-'+str(level)+'-final'

    lat = np.arange(55-lat_offset,55+lat_offset+0.5,0.5)[::-1]
    lon = np.arange(240-lon_offset,240+lon_offset+0.5,0.5)
    colors = ["#0072B2","#CC79A7","#D2D2D2","#4268B3"]

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
    elif res == 168:
        t_step.append(slice(np.array(int((7*24)*1e9*60*60),dtype='timedelta64[ns]'),  # day 7-14
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
    clim_dists(obs_full.sel(index=train_period))
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
    if seas == 'MAM':
        indices=np.arange(N_nodes).reshape(Ny,Nx).flatten()
    elif seas == 'SON':
        indices = np.concatenate([np.arange(16).reshape(8,2),np.arange(16,32).reshape(8,2)],axis=1).flatten()

    with open('distributions-'+title[4:]+'.pkl','rb') as handle:
        distributions = pickle.load(handle)

    #plot_som(Nx, Ny, som.z_raw, indices)
    #plot_distributions_6x4(distributions)


    # era data (used for validation - finding bad nodes)
    if level == 850:
        era = xr.open_dataset('era-850-2009-2022.nc').sel(latitude=lat,longitude=lon-360)
    else:
        era = xr.open_dataset('era-2009-2022-a.nc').sel(latitude=lat,longitude=lon-360,level=level)
    erav = era.sel(time=val_period)
    era = era.sel(time=train_period)
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
    erav = resample_mean(erav,'era',6)
    erav = erav.sel(time=obs_val.index.values)  # only keep indices with valid obs

    # do forecasting, stats
    bad_nodes = era_stats(agg)

    bad_r_som = []
    bad_brier_som = []

    stats_all = forecast()
    print('Doing clim stats')
    stats_all = clim_stats(stats_all)

    print('Getting gefs stats...')
    bad_r_gefs = []
    bad_brier_gefs = []
    gefs = xr.open_dataset('~/Nextcloud/thesis/gefs-wind-all-interpolated.nc').sel(time=test_period_gefs)
    if res > 24:
        gefs = gefs.sel(step=slice(t_step[0].start,t_step[-1].stop))
    gefs = resample_mean(gefs,'gefs-wind',res) # resampled in the 'step' dimension
    stats_all = gefs_stats(gefs,stats_all)

    do_plots(stats_all)

    #plot_discarded(discarded,t_step_plot)

    #box_plots(crps,crps_clim,crps_gefs,t_step_plot)
   

    print('Done')


    
