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
from scipy import signal
from sklearn.metrics import brier_score_loss
import scipy.stats as stats
from datetime import datetime
import glob
import pandas as pd
import warnings
warnings.filterwarnings("ignore")



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

    return clim_prob, clim_det


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
    for f in write_file_list:
        with open(f,'a') as file:
            file.write(','+str(TE)+','+str(QE)+','+str(EV)+','+str(PF)+','+str(WSS)
                +','+str(ks_sig_frac)+','+str(dist_spread)+','+str(dist_std))
        
    return None


def gefs_stats(bad_nodes,f):
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
    obsv_loc = obs_val.copy(deep=True)  # does not include nan
    res_str = str(res_obs) + 'h'
    files = glob.glob('data/z-gefs/gefs-z-*')
    print(len(files), ' members to process ' ,datetime.now())
    time_axis = pd.date_range(val_period_gefs.start,val_period_gefs.stop,freq=res_str)

    dist_xr_list = []
    for x in range(len(bad_nodes)):
        bmus_mems = np.empty((len(time_axis),len(files),len(leads)))

        # bmus mems is shape (days, members, lead times)
        bmus_mems.fill(np.nan)

        dist_xr = xr.DataArray(data=bmus_mems,dims=['index','member','leadtime'],
            coords=dict(index=time_axis,member=range(1,len(files)+1),leadtime=leads))
        
        dist_xr_list.append(dist_xr)
    
    for gefs_file in files:  # each member
        mem=int(gefs_file[-5:-3])
        # open all of them and calculate best bmus, to get memory out of the way
        # save in a dataset so each validation date has a corresponding distribution
        current_gefs = xr.open_dataset(gefs_file).sel(time=val_period_gefs,isobaricInhPa=level)
        try:
            current_gefs = current_gefs.drop_vars(['valid_time'])
        except:
            None
        current_gefs['longitude'] = current_gefs['longitude'] - 360
        current_gefs = resample_mean(current_gefs.sel(latitude=lat,longitude=lon-360),'gefs',6) # resampled in the 'step' dimension

        for tt,t in enumerate(t_step):  # for each lead time
            # filter gefs by date so that it matches obs date
            if type(t) == slice:
                gefs_step = current_gefs.sel(step=t)  
                date = slice(obsv_loc.index.values[0]-t.start,obsv_loc.index.values[-1]-t.start)
                t = t.start

            else:
                gefs_step = current_gefs.sel(step=[t],method='nearest')
                date = slice(obsv_loc.index.values[0]-t,obsv_loc.index.values[-1]-t)

            gefs_step = gefs_step.sel(time=date)  # only keep indices with valid obs

            # get bmu
            # arrays at this point include nan dates
            # every lead time for every day of every member has a bmu
            gefs_step=gefs_step.transpose("time","step","latitude","longitude")  # put time first so that can enumerate thru
            for kk, st in enumerate(gefs_step):  # for each date
                date = st.time.values
                if date + t in obsv_loc.index.values:
                    st = st - clim.sel(dayofyear=pd.to_datetime(date+t).day_of_year)  # ANOMOLAY STUFF
                    for ii, gh in enumerate(st.gh):  # for each step
                        mem_arr = np.reshape(gh.to_numpy(),(gh.latitude.shape[0]*gh.longitude.shape[0]))
                        BMU = np.argmin(np.linalg.norm(mem_arr - som.z_raw, axis=1))
                        l = leads[int(tt*(res/6)+ii)]

                        for x in range(len(dist_xr_list)):
                            if np.isnan(gh.any()) or BMU in bad_nodes[x]:  # don't count bad nodes
                                dist_xr_list[x].loc[dict(index=date,member=mem,leadtime=l)] = np.nan
                            else:
                                dist_xr_list[x].loc[dict(index=date,member=mem,leadtime=l)] = BMU
    print('stats time ' ,datetime.now())
    for x in range(len(dist_xr_list)):
        dist_xr = dist_xr_list[x].dropna(dim='index',how='all')

        # get stats for each lead time
        r_list = []
        crps_list = []
        bs_list = []
        mae_list = []
        bias_list = []
        crpss_list = []
        discarded = []
        D_list = []
        ks_list=[]
        PF_list = []
        for tt,t in enumerate(t_step):  # for each lead time
            obsv_loc = obs_val.copy(deep=True)  # reset

            dist_arr = dist_xr.sel(leadtime=t)
            nan_count = 0
            smth = []

            for kk, ob in enumerate(obsv_loc.Wind):  # for each testing obs
                dist_list=[]

                if np.isnan(ob.values):  # here we skip nan dates
                    continue

                # get bmus from this period and add the distributions
                date = ob.index.values
                if res > 6:
                    for c in dist_arr.sel(index=date-t.start).values.T.squeeze():  # each date in here will have a list of members?
                        if np.count_nonzero(np.isnan(c)) > 0.5*len(dist_xr.member):  # if many of the members are bad nodes
                            dist_list.append([np.nan])
                        else:
                            dist_list.append(np.concatenate([distributions[int(e)] for e in c if ~np.isnan(e)]))
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
            if len(smth) == 0:
                
                r_list.append(np.nan)
                crps_list.append(np.nan)
                bs_list.append(np.nan)
                mae_list.append(np.nan)
                bias_list.append(np.nan)
                crpss_list.append(np.nan)
                discarded.append(np.nan)
                D_list.append(np.nan)
                ks_list.append(np.nan)
                PF_list.append(np.nan)
                continue
            else:
                dists_time = stack_uneven(smth)
            obsv_cut = obsv_loc.dropna(dim='index',how='all')
            if res < 24:
                obsv_cut = obsv_cut.where(obsv_cut['index.hour']==pd.to_datetime(t.astype('datetime64')).hour, drop=True)

            if len(dists_time) != len(obsv_cut.index):
                print('bad things!!!')
                os.abort()
            discarded.append(nan_count / (nan_count + len(dists_time)))
            
            # remake some of the map stats based on the new distributions
            # pseudo-F and K-S both tell us about the uniqueness of the distributions
            dist_means = np.nanmean(dists_time,axis=1)
            total_mean = np.nanmean(dist_means)
            BSS_nodes = np.empty(len(dist_means))
            crps_som = np.empty((len(dist_means)))  # (nodes, forecasts)
            crps_som.fill(np.nan)
            crps_clim = np.empty((len(dist_means)))
            crps_clim.fill(np.nan)

            WSS = np.nansum(np.square(dists_time - np.vstack(dist_means)))
            TSS = np.sum(np.square(obs - total_mean))

            sig_count = 0
            m = len(obs_train.Wind.values)
            for i in range(len(dist_means)):
                BSS_nodes[i] = np.square(dist_means[i] - total_mean)
                # K-S test
                n = dists_time[i,:].shape[0]

                crit = 1.63*np.sqrt((n+m)/(n*m))  # 1.36 is for 95% confidence, 1.07 for 80 1.22 for 90 1.52 for 98 1.63 for 99
                ks = stats.ks_2samp(dists_time[i,:], obs_train.Wind.values)  # rejection means the distirbutions are different
                if ks.statistic > crit and ks.pvalue < 0.05:  # rejection of null
                    sig_count +=1

                ob1 = obsv_cut.isel(index=i).Wind  # wind observation for this date
                date = pd.to_datetime(ob1.index.values)

                crps_som[i] =ps.crps_ensemble(ob1, dists_time[i,:])
                crps_clim[i] =ps.crps_ensemble(ob1, clim_prob.sel(index=clim_prob.index.hour==gh.time.dt.hour).Wind)
            crpss = np.nanmean(1-crps_som/crps_clim)

            ks_sig_frac = sig_count / len(dist_means)  # perentage of nodes that have significantly different distribution
            PF = (np.sum(BSS_nodes)/(len(dist_means)-1)) / (WSS/(m-len(dist_means)))  # pseudo-F statistic

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
            R_list = []

            for j,d in enumerate(dists_time):  # for each distribution
                a = d[~np.isnan(d)]
                u_list = []
                for i,ii in enumerate(dists_time):  # for each distribution again to compare lol
                    if j == i:
                        continue
                    b = ii[~np.isnan(ii)]
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

            D = 0.5*(stats.kendalltau(obsv_cut.Wind.values,R_list)[0] + 1)  # generalized discrimination score (Weigel&Mason 2011)

            # windy vs not windy
            splits = [25,50,75,90,95]
            bss = []
            for percentile in splits:
                split = np.percentile(obs, percentile)
                target = np.array(obsv_cut.Wind > split)  # boolean array of obs
                prob = [np.sum(d>split)/len(d[~np.isnan(d)]) for d in dists_time]  # probabilities of being above threshold in each distribution
                prob_clim = np.sum([d>split for d in obs])/len(obs)
                brier = brier_score_loss(target, prob)
                brier_clim = brier_score_loss(target, np.repeat(prob_clim,len(target)))
                bss.append(1 - brier/brier_clim)
            r_list.append(r_value)
            crps_list.append(np.nanmean(crps_som))
            bs_list.append(bss)
            bias_list.append(bias)
            mae_list.append(mae)
            crpss_list.append(crpss)
            D_list.append(D)
            ks_list.append(ks_sig_frac)
            PF_list.append(PF)
        with open(f[x],'a') as file:
            file.write(','+str(PF_list) +','+str(ks_list)+','+str(crpss_list)+','+str(r_list)+
                        ','+str(mae_list)+','+str(bias_list)+','+str(D_list)+','+str(bs_list)+','+str(discarded))

    return None


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

    bad_nodes = [[],[],[]]
    for j,d in enumerate(distributions):  # for each node
        # calculate total crpss for each node
        crpss = np.nanmean(1-crps_som[j,:]/crps_clim[j,:])
        if crpss < -0.5:
            bad_nodes[0].append(j)
        if crpss < 0.01:  # flag indices of 'bad' nodes
            bad_nodes[1].append(j)
        if crpss < 0.05:
            bad_nodes[2].append(j)

    return bad_nodes



if __name__ ==  "__main__":
    # setup
    print('starting',datetime.now())
    x_to_try = np.arange(2,17)
    sizes = []
    for x in x_to_try:
        for y in range(1,x+1):
            if x*y < 37:
                sizes.append((x,y))

    seas = 'JJA'
    title = '24h-summer-1000'

    lat_dif = [9,11]  # domain size (degrees on each side of the center)
    lon_dif = [16,20]
    res = 6  # time resolution of map in hours, always 6
    res_obs = 24

    anomaly = True  # get rid of seasonal anomaly
    train_period=slice("2009-10-01","2020-09-23")
    val_period = slice("2020-10-01","2022-03-31")
    val_period_gefs = slice("2020-09-24","2022-03-24")
    level = 1000

    # this is for lead time. if res > 6H, t_step is slice so that we pick up multiple z forecasts
    t_step = []
    for d in range(6,15): # each day in week 2
        if res == 24:
            t_step.append(slice(np.array(int((d*24+3)*1e9*60*60),dtype='timedelta64[ns]'),  # adding 3 hours for index shift
                                np.array(int(((d+0.9)*24+3)*1e9*60*60),dtype='timedelta64[ns]')))
        elif res == 6:
            t_step.append(np.array(int(d*24*1e9*60*60),dtype='timedelta64[ns]'))
            t_step.append(np.array(int((d*24+6)*1e9*60*60),dtype='timedelta64[ns]'))
            t_step.append(np.array(int((d*24+12)*1e9*60*60),dtype='timedelta64[ns]'))
            t_step.append(np.array(int((d*24+18)*1e9*60*60),dtype='timedelta64[ns]'))
        else:
            print('only configured for 6h and 24 resolutions right now')
            raise ValueError

    print('Loading data...')

    obs_full = xr.open_dataset('~/Nextcloud/thesis/bm_cleaned_all.nc')
    obs_full = resample_mean(obs_full,'obs',res)
    obs_full = obs_full.sel(index=obs_full.index.dt.season==seas)
    obs_train = obs_full.sel(index=train_period)
    clim_prob, clim_det = get_climo_det(obs_train)  # for both datasets need to select the intended hour

    obs_val = obs_full.sel(index=val_period)
    obs_val = obs_val.dropna(dim='index')

    obs_full = None
    write_file_list = ['stats-'+title+'-0.txt','stats-'+title+'-1.txt',
                       'stats-'+title+'-2.txt']
    for f in write_file_list:
        with open(f, 'w') as file:
            file.write('Nx,Ny,Nnodes,lat,lon,TE,QE,EV,PF,WSS,KS-frac,range,std,PF-gefs,KS-gefs'+
                    ',CRPSS,R-gefs,mae-gefs,bias-gefs,D-gefs,bss-gefs,frac-discarded')
        
    for dom in range(len(lat_dif)):  # for each domain size
        lat_offset = lat_dif[dom]
        lon_offset = lon_dif[dom]
        lat = np.arange(55-lat_offset,55+lat_offset+0.5,0.5)[::-1]
        lon = np.arange(240-lon_offset,240+lon_offset+0.5,0.5)

        if level == 850:
            era = xr.open_dataset('era-850-2009-2022.nc').sel(latitude=lat,longitude=lon-360)
        else:
            era = xr.open_dataset('era-2009-2022-a.nc').sel(latitude=lat,longitude=lon-360,level=level)
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
        erav = erav.sel(time=obs_val.index.values)  # only keep indices with valid obs

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
            for f in write_file_list:
                with open(f,'a') as file:
                    file.write('\n'+str(Nx)+','+str(Ny)+','+str(N_nodes)+','+str(lat_offset)+','+str(lon_offset))

            print('\n training map...',Nx,'x',Ny,flush=True)
            som = train_som(era)
            
            indices = np.arange(N_nodes).reshape(Nx,Ny).T.flatten()
            bmus = BMUs(som)  # the nodes that each gh best matches
            distributions = wind_distributions(bmus)
            freq = BMU_frequency(som)  # frequency of each node

            # map-based validation statistics
            map_stats()
 
            bad_nodes = era_stats()  # nested list. goes from no filter to aggressive
            good_ind = []
            for b in range(len(bad_nodes)):
                if len(bad_nodes[b]) == N_nodes:  # if all nodes are bad
                    print('all nodes bad')
                else:
                    good_ind.append(b)
            if len(good_ind) == 0:
                print('all version of this map bad')
                continue
            else:
                # validation statistics for gefs
                print('getting gefs stuff')
                gefs_stats([bad_nodes[g] for g in good_ind],[write_file_list[g] for g in good_ind])

        toc = time.perf_counter()
        print('Done that domain size',dom+1,f'/2 in {(toc - tic)/60:0.2f} minutes.')
    print('Done',datetime.now())

        
