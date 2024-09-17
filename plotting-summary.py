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
import matplotlib.ticker as mticker
from cmcrameri import cm
import matplotlib.cm as comap
import glob
import sys
import warnings
warnings.filterwarnings("ignore")
import math
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

            stats_xr['width'].loc[dict(index=date,leadtime=t)]= np.max(dists_nonan) - np.min(dists_nonan)
            w95=np.percentile(dists_nonan, 95)
            w5=np.percentile(dists_nonan, 5)
            w75=np.percentile(dists_nonan, 75)
            w25=np.percentile(dists_nonan, 25)
            stats_xr['width90'].loc[dict(index=date,leadtime=t)]= w95-w5
            stats_xr['width50'].loc[dict(index=date,leadtime=t)]= w75-w25

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

        #ds['width_clim'].loc[dict(index=gh)]= np.max(dists_nonan) - np.min(dists_nonan)
        #w95=np.percentile(dists_nonan, 95)
        #w5=np.percentile(dists_nonan, 5)
        w75=np.percentile(dists_nonan, 75)
        w25=np.percentile(dists_nonan, 25)
        ds['width50_clim'].loc[dict(index=gh)]= w75-w25

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
                #st['prob_gefs'].loc[dict(index=obs_date,leadtime=t)]= sum(v < ob for v in dists_nonan) / len(dists_nonan)

                #st['width_gefs'].loc[dict(index=obs_date,leadtime=t)]= np.max(dists_nonan) - np.min(dists_nonan)
                # w95=np.percentile(dists_nonan, 95)
                # w5=np.percentile(dists_nonan, 5)
                w75=np.percentile(dists_nonan, 75)
                w25=np.percentile(dists_nonan, 25)
                st['width50_gefs'].loc[dict(index=obs_date,leadtime=t)]= w75-w25
        mae_list.append(np.mean(mae))
        bias_list.append(np.mean(bias))
        crps_list.append(np.array(crps))
        number.append(len(mae))

        # slope, intercept, r_gefs, p_gefs, std_err = stats.linregress(g_time_series, ob_time_series)
        # if p_gefs> 0.05:
        #     bad_r_gefs.append(tt)
        # r_list.append(r_gefs)
        
        # splits = [50]
        # bss = []
        # for percentile in splits:
        #     split = np.percentile(obs_train.Wind.dropna(dim='index'), percentile)
        #     target = np.array(ob_time_series > split)  # boolean array of obs
        #     prob = [np.sum(d>split)/len(d[~np.isnan(d)]) for d in g_time_series_members]  # probabilities of being above threshold in each distribution
        #     s, pvalue = dstats.sign_test((prob-target)**2,0.25)
        #     brier = brier_score_loss(target, prob)
        #     if pvalue > 0.05:
        #         bad_brier_gefs.append(tt)

        # bs50_list.append(brier)

    # st['r_gefs'] = (['leadtime'],r_list)
    # st['mae_gefs'] = (['leadtime'],mae_list)
    # st['bs50_gefs'] = (['leadtime'],bs50_list)
    # st['number_gefs'] = (['leadtime'],number)

    return st
            

def plot_error(som_error,gefs_error, ltimes):
    ltimes = np.array(ltimes,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)
    linestyles = ['solid','dashed','dotted']
    cols = ["#1923B5","#23DA42","#E240AB","#844105"]

    fig, ax = plt.subplots(figsize=(7,5))

    for i in range(0,1):  # for each res
        err = som_error[i]
        for j in range(4): # for each season
            err_seas = err.isel(season=j)
            ax.plot(np.arange(6,15,res_list[i]/24), np.nanmedian(err_seas,axis=0),
                     label='SOM-'+seasons[j],zorder=3,linestyle=linestyles[i],c=cols[j],lw=1.2)
            
    for i in range(0,1):  # for each res
        err = gefs_error[i]
        for j in range(4): # for each season
            err_seas = err.isel(season=j)
            ax.plot(np.arange(6,15,res_list[i]/24), np.nanmedian(err_seas,axis=0),
                     label='GEFS-'+seasons[j],zorder=3,linestyle=linestyles[i+1],c=cols[j],lw=1,alpha=0.1)

    ax.axhline(0, label='Climatology',c='black',zorder=1)
    ax.set_xlabel('Lead time (days)',fontsize=14)
    ax.set_ylabel('CRPSS',fontsize=14)
    ax.set_xticks(range(6,16))
    ax.xaxis.grid(True,which='both',linestyle='--')
    ax.legend(bbox_to_anchor=(1.5, 1.5),fontsize=12)
    ax.tick_params(axis='both',labelsize=12,pad=1)
    plt.tight_layout()
    plt.savefig('plots/summary-CRPSS.pdf',bbox_inches='tight')

    return None

def plot_time_error(som_error, ltimes):
    ltimes = np.array(ltimes,dtype=np.float64)
    ltimes = ltimes / (24*60*60*1e9)
    linestyles = ['solid',(5,(10,3)),'dashed','dotted','dashdot']
    cols = ["#1923B5","#23DA42","#E240AB","#844105"]

    fig, ax = plt.subplots(figsize=(7,4))

    for i in range(0,5):  # for each res
        err = som_error[i]
        for j in range(0,1): # for each season
            err_seas = err.isel(season=j)
            if i == 4:  # 7-day
                ax.plot([7,14],[np.nanmedian(err_seas,axis=0),np.nanmedian(err_seas,axis=0)],
                        label=str(res_list[i])+'h',zorder=3,linestyle=linestyles[i],c=cols[j],lw=1.2)
            elif i == 3:  # 48-h
                ax.plot(np.arange(7,15,res_list[i]/24), np.nanmedian(err_seas,axis=0),
                        label=str(res_list[i])+'h',zorder=3,linestyle=linestyles[i],c=cols[j],lw=1.2)
            else:
                ax.plot(np.arange(6,15,res_list[i]/24), np.nanmedian(err_seas,axis=0),
                        label=str(res_list[i])+'h',zorder=3,linestyle=linestyles[i],c=cols[j],lw=1.2)

    ax.fill_between([6,15],3.21,4,label='Climatology',color='grey',alpha=0.3,zorder=1)
    ax.set_xlabel('Lead time (days)',fontsize=14)
    ax.set_ylabel('Width (m/s)',fontsize=14)
    ax.set_xticks(range(6,16))
    ax.xaxis.grid(True,which='both',linestyle='--')
    ax.legend(fontsize=12,borderaxespad=0.1,loc='upper right')
    ax.tick_params(axis='both',labelsize=12,pad=1)
    plt.tight_layout()
    plt.savefig('plots/summary-width-DJF.pdf',bbox_inches='tight')

    return None


def width_plot(crpss_list,width_list,cli):
    cols = ["#1923B5","#23DA42","#E240AB","#844105"]
    t = [6,12,24,48,168]

    fig, ax = plt.subplots(1,1,figsize=(7,4))

    ax.axhline(y=0,lw=2,label='Climatology',c='black')
    for i in range(4): # each season
        # plot each time res in a line
        ax.plot([np.mean(np.nanmedian(crpss_list[0].isel(season=i),axis=0)),
            np.mean(np.nanmedian(crpss_list[1].isel(season=i),axis=0)),
            np.mean(np.nanmedian(crpss_list[2].isel(season=i),axis=0)),
            np.mean(np.nanmedian(crpss_list[3].isel(season=i),axis=0)),
            np.mean(np.nanmedian(crpss_list[4].isel(season=i),axis=0))],c=cols[i],label=seasons[i])
   
    ax.set_xlabel('Temporal Resolution (h)',fontsize=14)
    ax.set_ylabel('CRPSS',fontsize=14)
    ax.set_xticks([0,1,2,3,4],t)
    ax.legend(fontsize=12,borderaxespad=0.1,loc='lower left')
    ax.tick_params(axis='both',labelsize=12,pad=1)
    ax.xaxis.grid(True,which='both',linestyle='--')
    plt.savefig('plots/summary-crpss-gefs.pdf',bbox_inches='tight')
        
    fig, ax = plt.subplots(1,1,figsize=(7,4))
    ax.plot([cli[0].mean(dim=['season','index']),cli[1].mean(dim=['season','index']),
               cli[2].mean(dim=['season','index']),cli[3].mean(dim=['season','index']),
               cli[4].mean(dim=['season','index'])],label='Climatology',c='black',lw=2)
    for i in range(4): # each season
        ax.plot([np.mean(np.nanmedian(width_list[0].isel(season=i),axis=0)),
            np.mean(np.nanmedian(width_list[1].isel(season=i),axis=0)),
            np.mean(np.nanmedian(width_list[2].isel(season=i),axis=0)),
            np.mean(np.nanmedian(width_list[3].isel(season=i),axis=0)),
            np.mean(np.nanmedian(width_list[3].isel(season=i),axis=0))],c=cols[i],label=seasons[i])
    ax.set_xlabel('Temporal Resolution (h)',fontsize=14)
    ax.set_ylabel('Width (m/s)',fontsize=14)
    ax.set_xticks([0,1,2,3,4],t)
    ax.legend(fontsize=12,borderaxespad=0.1,loc='lower left')
    ax.tick_params(axis='both',labelsize=12,pad=1)
    ax.xaxis.grid(True,which='both',linestyle='--')
    plt.savefig('plots/summary-width-gefs.pdf',bbox_inches='tight')
        
    return None



def do_plots(ds1):
    crpss_list = []
    width_list = []
    width_clim = []
    crpss_list_gefs = []
    width_list_gefs = []
    for x in range(len(ds1)):  # each resolution
        ds = ds1[x]
        res = res_list[x]
        crpss = 1 - ds['crps']/ds['crps_clim']
        crpss_gefs = 1 - ds['crps_gefs']/ds['crps_clim']

        bss50 = 1 - ds['bs50'] / 0.25

        bad_som_crpss_djf = []
        bad_som_crpss_mam = []
        bad_som_crpss_jja = []
        bad_som_crpss_son = []
        bad_list = [bad_som_crpss_djf,bad_som_crpss_mam,bad_som_crpss_jja,bad_som_crpss_son]

        for i in range(len(ds.leadtime)):
            l = ds.leadtime[i]
            h = pd.to_datetime(l.astype('datetime64[ns]').values).hour
            for j in range(len(ds.season)):
                #crpss
                s,p = dstats.sign_test(crpss.sel(leadtime=l).isel(season=j).dropna(dim='index'))
                if p > 0.05:
                    bad_list[j].append(i)

                # # mae
                # s,p = stats.mannwhitneyu(abs(ds['bias'].sel(leadtime=l,season=j).dropna(dim='index')),abs(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h).dropna(dim='index')))
                # if p > 0.05:
                #     bad_som_mae.append(i)

                # # bias
                # s,p = stats.mannwhitneyu(ds['bias'].sel(leadtime=l,season=j).dropna(dim='index'),ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h).dropna(dim='index'))
                # if p > 0.05:
                #     bad_som_bias.append(i)

                # # sharpness
                # s,p = dstats.sign_test(ds['width50'].sel(leadtime=l,season=j).dropna(dim='index'),ds['width50_clim'].isel(index=i%4).values)
                # if p > 0.05:
                #     bad_sharp_som.append(i)

        width_list.append(ds['width50'])
        crpss_list.append(crpss)
        width_list_gefs.append(ds['width50_gefs'])
        crpss_list_gefs.append(crpss_gefs)
        width_clim.append(ds['width50_clim'])

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

            # djf
            b = round(np.nanmedian(crpss.isel(season=0).sel(leadtime=l)),2)
            if i in bad_som_crpss_djf:
                st += str(b) + ' &'
            elif b > 0:
                st += '\\cellcolor{green!25}\\textbf{' + str(b) + '} &'
            else:
                st += '\\textbf{' + str(b) + '} &'

            # mam
            b = round(np.nanmedian(crpss.isel(season=1).sel(leadtime=l)),2)
            if i in bad_som_crpss_mam:
                st += str(b) + ' &'
            elif b > 0:
                st += '\\cellcolor{green!25}\\textbf{' + str(b) + '} &'
            else:
                st += '\\textbf{' + str(b) + '} &'

            # jja
            b = round(np.nanmedian(crpss.isel(season=2).sel(leadtime=l)),2)
            if i in bad_som_crpss_jja:
                st += str(b) + ' &'
            elif b > 0:
                st += '\\cellcolor{green!25}\\textbf{' + str(b) + '} &'
            else:
                st += '\\textbf{' + str(b) + '} &'

            # son
            b = round(np.nanmedian(crpss.isel(season=3).sel(leadtime=l)),2)
            if i in bad_som_crpss_son:
                st += str(b)
            elif b > 0:
                st += '\\cellcolor{green!25}\\textbf{' + str(b) + '}'
            else:
                st += '\\textbf{' + str(b) + '}'

            st +=r' \\'
            st_gefs += r'\\'
            if res == 12 and i%2 == 0:
                st += r'\cline{2-6}'
                st_gefs += r'\cline{2-7}'
            elif res == 12 and i%2 != 0:
                st += '\n'+ r'\hline'
                st_gefs += '\n' + r'\hline' 

            elif res == 6 and (i%4 == 0 or (i-1)%4==0 or (i-2)%4==0):
                st += r' \cline{2-6}'
                st_gefs += r' \cline{2-7}' 
            elif res == 6 and (i-3)%4 == 0:
                st += '\n'+ r'\hline'
                st_gefs += '\n' + r'\hline' 
            
            else:
                st += '\n' + r'\hline'
                st_gefs += '\n' + r'\hline'

            full_st+= (st+'\n')
            full_st_gefs+= (st_gefs+'\n')
        # full_st = full_st[:-8]
        # full_st += '\n'+r'\Xhline{2.5\arrayrulewidth}' + '\n'
        # full_st_gefs = full_st_gefs[:-8]
        # full_st_gefs += '\n'+r'\Xhline{2.5\arrayrulewidth}' + '\n'

        # if res >= 24:
        #     full_st += 'Clim &'+(str(round(np.nanmean(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==0)),2))
        #                 + ' & ' + str(round(np.nanmean(abs(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==0))),2))
        #                 + r' & N/A & 0 & 0 & N/A\\'+'\n' + r'\hline')
        #     full_st_gefs += 'Clim &'+(str(round(np.nanmean(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==0)),2))
        #                 + ' & ' + str(round(np.nanmean(abs(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==0))),2))
        #                 + r' & N/A & 0 & 0 N/A \\'+'\n' + r'\hline')
        # else:
        #     full_st +='Clim '
        #     full_st_gefs += 'Clim '
        #     if res == 12:
        #         h_list = [0,12]
        #     elif res == 6:
        #         h_list = [0,6,12,18]
        #     for h in h_list:
        #         full_st += ' &'+str(h)+' &' + (str(round(np.nanmean(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h)),2))
        #                     + ' & ' + str(round(np.nanmean(abs(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h))),2))
        #                     + r' & N/A & 0 & 0 & N/A\\'+'\n' +  r' \cline{2-8}')
        #         full_st_gefs += ' &'+str(h)+' &' + (str(round(np.nanmean(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h)),2))
        #                     + ' & ' + str(round(np.nanmean(abs(ds['bias_clim'].sel(index=pd.to_datetime(ds.index).hour==h))),2))
        #                     + r' & N/A & 0 & 0 N/A \\'+'\n' +  r' \cline{2-7}')
        # print('SOM \n' + full_st)
        # full_st += '\n'+'\n'+str(ds['number'].isel(leadtime=0).values) + ' total forecast-obs pairs'

        # with open(('table-summary-'+str(res_list[x])+'h.txt'), 'w') as file:
        #     file.write(full_st)
    # with open(('table-'+seas+'-'+str(res)+'h-GEFS.txt'), 'w') as file:
    #     file.write(full_st_gefs)

    plot_error(crpss_list,crpss_list_gefs,t_step_plot)
    #plot_time_error(width_list,t_step_plot)
    #width_plot(crpss_list,width_list,width_clim)
    #width_plot(crpss_list_gefs,width_list_gefs,width_clim)

    return None



if __name__ == '__main__':
    res_list = [6]
    train_period=slice("2009-10-01","2020-09-23")
    val_period = slice("2020-10-01","2022-03-31")
    test_period_gefs = slice("2022-03-25","2024-03-25")  # 6 days earlier so that dates line up
    test_period = slice("2022-04-01","2024-04-01")
    lat_offset = 9
    lon_offset = 16
    lat = np.arange(55-lat_offset,55+lat_offset+0.5,0.5)[::-1]
    lon = np.arange(240-lon_offset,240+lon_offset+0.5,0.5)
    colors = ["#0072B2","#CC79A7","#BFBFBF","#4268B3"]
    seasons = ['DJF','MAM','JJA','SON']
    ds_list = []

    for res in res_list:
        res_str = str(res)+'h'
        t_step = []
        stats_list = []
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
        elif res == 168:
            t_step.append(slice(np.array(int((7*24)*1e9*60*60),dtype='timedelta64[ns]'),  # day 7-14
                                    np.array(int(((13+0.9)*24)*1e9*60*60),dtype='timedelta64[ns]')))
        else:
            print('Invalid resolution')
            raise ValueError
        
        t_step_plot = []
        if res > 6:
            for l in t_step:
                t_step_plot.append(l.start)
        else:
            t_step_plot = t_step


        for seas in seasons:
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

            do = False
            num_members = 15

            # setup
            N_nodes = Nx * Ny
            title = '24h-'+seas+'-'+str(level)+'-final'

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
            if seas == 'MAM':
                indices=np.arange(N_nodes).reshape(Ny,Nx).flatten()
            elif seas == 'SON':
                indices = np.concatenate([np.arange(16).reshape(8,2),np.arange(16,32).reshape(8,2)],axis=1).flatten()

            with open('distributions-'+title[4:]+'.pkl','rb') as handle:
                distributions = pickle.load(handle)

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

            gefs = xr.open_dataset('~/Nextcloud/thesis/gefs-wind-all-interpolated.nc').sel(time=test_period_gefs)
            if res > 24:
                gefs = gefs.sel(step=slice(t_step[0].start,t_step[-1].stop))
            gefs = resample_mean(gefs,'gefs-wind',res) # resampled in the 'step' dimension
            stats_all = gefs_stats(gefs,stats_all)

            stats_list.append(stats_all)
        ds_merged = xr.concat(stats_list,pd.Index(seasons,name='season'))
        ds_list.append(ds_merged)
        print('Done this res: ', res, 'hr')

    do_plots(ds_list)



        
