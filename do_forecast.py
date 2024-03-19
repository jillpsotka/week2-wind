import numpy as np
import pickle
import pandas as pd
from som_class import SOM, BMUs, BMU_frequency
from matplotlib import pyplot as plt
import xarray as xr
from resampling import gefs_reanalysis, dates_obs_gefs, resample_mean, low_pass_filter
from do_som import prep_data
import scipy.stats as stats
import properscoring as ps
import datetime
from scipy import signal


def get_deterministic_climo(ds):
    # get climo by doy

    # first rolling avgs to smooth a bit
    ds = ds.rolling(index=5,min_periods=3,center=True).mean().rolling(index=21,min_periods=11,center=True).mean()

    # then group by day
    ds = ds.groupby("index.dayofyear").mean(dim='index')

    # then low pass filter to smooth again
    cutoff=0.08
    b, a = signal.butter(5, cutoff, btype='lowpass')

    dUfilt = signal.filtfilt(b, a, ds.Wind.values,axis=0)
    ds.Wind.values = dUfilt

    return ds


if __name__ ==  "__main__":

    res = 24
    dates_train = slice("2009-10-01", "2020-09-30")
    dates_test = slice("2020-10-01","2021-12-31")
    dates_obs = slice("2020-10-01","2021-12-31")
    anomaly = True
    number=600

    # let's start with day 7 00:00
    t = np.array(int(7*24*1e9*60*60),dtype='timedelta64[ns]')

    print('Opening things...')
    # som results
    with open('distributions-'+str(res)+'h-2x2-anomalies-all.pkl','rb') as handle:
        dist = pickle.load(handle)  # distributions associated with each map node
    with open('trained-map-'+str(res)+'h-2x2-anomalies-all.pkl', 'rb') as handle:
        som = pickle.load(handle)

    obs =  xr.open_dataset('data/bm_cleaned_all.nc')
    obs_test = low_pass_filter(obs.sel(index=dates_obs),'obs',res)  # test dates and resample
    obs_train = low_pass_filter(obs.sel(index=dates_train),'obs',res)
    clim_det = get_deterministic_climo(obs_train)
    obs_train = obs_train.Wind.to_numpy()

    #z_forecast = xr.open_dataset('data/gefs-z-2020-09-24-2023-12-31-0.nc').sel(time=dates_test)
    #z_forecast = low_pass_filter(z_forecast,'gefs',res).sel(step=t)  # resampled, day 7 00:00
    #z_forecast['longitude'] = z_forecast['longitude'] - 360  # so that it matches era

    era = xr.open_dataset('era-2009-2022.nc')#.sel(time=dates_train)
    era = low_pass_filter(era,'era',res)

    if anomaly:
        clim = era.sel(time=dates_train).groupby("time.dayofyear").mean(dim=["time"])
        clim_concat = xr.concat([clim.isel(dayofyear=slice(330,366)),clim,clim.isel(dayofyear=slice(0,36))],dim='dayofyear')
        cutoff=0.03
        b, a = signal.butter(5, cutoff, btype='lowpass')
        dUfilt = signal.filtfilt(b, a, clim_concat.gh.values,axis=0)
        dUfilt = dUfilt[36:-36,:,:]
        clim.gh.values = dUfilt
        z_forecast = era.sel(time=dates_test).groupby("time.dayofyear") - clim

    #wind_forecast = xr.open_dataset('data/gefs-wind-2020-12-17-2023-12-31-0-interpolated.nc').sel(time=dates_test,step=t)

    print('Calculating forecast...')
    # calculate which node each forecast belongs to
    BMUs = np.zeros(len(z_forecast.time), dtype='int')

    for kk, ob in enumerate(z_forecast.gh):
        ob = np.reshape(ob.to_numpy(),(ob.latitude.shape[0]*ob.longitude.shape[0]))
        BMU = np.argmin(np.linalg.norm(ob - som.z_raw[:,:-number], axis=1))
        BMUs[kk] = BMU

    print('Doing stats....')
    # lets put together all the data for each time step and calculate the score!!!

    #crps
    crps_som = np.empty((len(dist),BMUs.shape[0]))  # (nodes, forecasts)
    crps_som.fill(np.nan)
    crps_clim = np.empty((len(dist),BMUs.shape[0]))
    crps_clim.fill(np.nan)
    #crps_wind = np.empty((len(dist),BMUs.shape[0]))
    #crps_wind.fill(np.nan)
    obs_node = np.empty((len(dist),BMUs.shape[0]))
    obs_node.fill(np.nan)

    #mae
    mae_clim = np.empty((len(dist),BMUs.shape[0]))
    mae_clim.fill(np.nan)
    mae_som_mean = np.empty((len(dist),BMUs.shape[0]))
    mae_som_mean.fill(np.nan)
    mae_som_median = np.empty((len(dist),BMUs.shape[0]))
    mae_som_median.fill(np.nan)

    #rmse
    rmse_clim = np.empty((len(dist),BMUs.shape[0]))
    rmse_clim.fill(np.nan)
    rmse_som_mean = np.empty((len(dist),BMUs.shape[0]))
    rmse_som_mean.fill(np.nan)
    rmse_som_median = np.empty((len(dist),BMUs.shape[0]))
    rmse_som_median.fill(np.nan)

    #bias
    bias_clim = np.empty((len(dist),BMUs.shape[0]))
    bias_clim.fill(np.nan)
    bias_som_mean = np.empty((len(dist),BMUs.shape[0]))
    bias_som_mean.fill(np.nan)
    bias_som_median = np.empty((len(dist),BMUs.shape[0]))
    bias_som_median.fill(np.nan)

    for i in range(BMUs.shape[0]):  # for each test date
        BMU = BMUs[i]
        dist_forecast = dist[BMU]
        dist_mean = np.mean(dist_forecast)
        dist_median = np.median(dist_forecast)
        date_init = z_forecast.isel(time=i).time.values
        date_forecast = date_init# + t
        month = date_forecast.astype('datetime64[M]').astype(int) % 12 + 1
        day = (date_forecast.astype('datetime64[D]') - date_forecast.astype('datetime64[M]')).astype(int) + 1
        #wind = wind_forecast.sel(time=date_init).wind.values
        clim = clim_det.sel(dayofyear=day).Wind.values

        ob = obs_test.sel(index=date_forecast).Wind.values
        obs_node[BMU,i] = ob
        crps_som[BMU,i] =(ps.crps_ensemble(ob, dist_forecast))
        crps_clim[BMU,i] =(ps.crps_ensemble(ob, obs_train))
        #crps_wind[BMU,i] =(ps.crps_ensemble(ob, wind))
        mae_clim[BMU,i] = np.abs(clim - ob)
        mae_som_mean[BMU,i] = np.abs(dist_mean - ob)
        mae_som_median[BMU,i] = np.abs(dist_median - ob)
        rmse_clim[BMU,i] = np.square(clim - ob)
        rmse_som_mean[BMU,i] = np.square(dist_mean - ob)
        rmse_som_median[BMU,i] = np.square(dist_median - ob)
        bias_clim[BMU,i] = clim - ob
        bias_som_mean[BMU,i] = dist_mean - ob
        bias_som_median[BMU,i] = dist_median - ob


    for i in range(crps_som.shape[0]):
        
        print('crpss',np.nanmean(1-crps_som[i,:]/crps_clim[i,:])) 
        print('mae clim',np.nanmean(mae_clim[i,:]))
        print('mae mean of som dist',np.nanmean(mae_som_mean[i,:]))
        print('mae median of som dist',np.nanmean(mae_som_median[i,:]))
        print('rmse clim',np.sqrt(np.nanmean(rmse_clim[i,:])))
        print('rmse mean of som dist',np.sqrt(np.nanmean(rmse_som_mean[i,:])))
        print('rmse median of som dist',np.sqrt(np.nanmean(rmse_som_median[i,:])))
        print('bias clim',np.nanmean(bias_clim[i,:]))
        print('bias mean of som dist',np.nanmean(bias_som_mean[i,:]))
        print('bias median of som dist',np.nanmean(bias_som_median[i,:]),'\n')


    print('done')


    