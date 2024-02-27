import numpy as np
import pickle
import pandas as pd
from fitter import Fitter, get_common_distributions, get_distributions
from som_class import SOM, BMUs, BMU_frequency
from matplotlib import pyplot as plt
import xarray as xr
from resampling import gefs_reanalysis, cleaned_obs, dates_obs_gefs, resample_mean
from do_som import prep_data
import scipy.stats as stats
import properscoring as ps
import datetime




def fit_hist(ds):
    # actually, maybe dont want to fit. can just use the distribution of obs?
    f = Fitter(ds, distributions=['beta','geninvgauss','chi2','ncx2','lognorm','nakagami'])
    f.fit()
    print(f.summary())

    dist = stats.nakagami(nu=0.765176, loc=1.6762, scale=4.758206) # from fitter
    dist.cdf(5) # this will give % of obs under 5m/s


    return None


def obs_cdf(x_arr):
    val = np.empty_like(x_arr)
    for i in range(x_arr.shape[0]):
        dist1 = np.argwhere(obs<=x_arr[i])
        val[i] = len(dist1)/len(obs)

    return val




if __name__ ==  "__main__":
    # obs = cleaned_obs(res=6)
    # gefs = xr.open_dataset('gh-reanalysis-all-2012-2019.nc').sel(isobaricInhPa=500)
    # gefs = gefs_reanalysis(gefs)
    # obs, gefs = dates_obs_gefs(obs, gefs)
    # obs, gefs = prep_data(gefs, obs)


    with open('distributions-obs.pkl','rb') as handle:
        obs_train = pickle.load(handle)
    with open('distributions-6h-2x2-anomalies-all.pkl','rb') as handle:
        dist = pickle.load(handle)
    with open('trained-map-6h-2x2-anomalies-all.pkl', 'rb') as handle:
        som = pickle.load(handle)

    # open all the things and choose dates
    # let's start with day 7 00:00
    t = np.array(int(7*24*1e9*60*60),dtype='timedelta64[ns]')
    dates = slice("2022-01-01", "2023-01-01")
    #climo = xr.open_dataset('data/climo-6h.nc')  # split into morning, day, evening, night
    obs_test = cleaned_obs(dates,6)

    dates = slice("2021-12-25","2022-12-25")
    z_forecast = xr.open_dataset('data/gefs-z-gh-2021-04-18-2023-02-18-0.nc').sel(time=dates)
    z_forecast = resample_mean(z_forecast,'gefs').sel(step=t)  # resampled because 3-hourly, day 7 00:00

    anomaly = True  # get rid of seasonal anomaly using 30-day rolling avg
    if anomaly:
        gefs = xr.open_dataset('era-reanalysis-2012-2022-6h.nc')
        gefs['longitude'] = gefs['longitude'] + 360
        gefs_smoothed = gefs.rolling(time=124,center=True).mean()
        clim = gefs_smoothed.groupby("time.dayofyear").mean(dim=["time"])
        z_forecast = z_forecast.groupby("time.dayofyear") - clim

    wind_forecast = xr.open_dataset('gefs-wind-2020-12-17-2023-12-31-interpolated-0.nc').sel(time=dates,step=t)

    # calculate which node each forecast belongs to
    BMUs = np.zeros(len(z_forecast.time), dtype='int')

    for kk, ob in enumerate(z_forecast.gh):
        ob = np.reshape(ob.to_numpy(),(ob.latitude.shape[0]*ob.longitude.shape[0]))
        BMU = np.argmin(np.linalg.norm(ob - som.z_raw, axis=1))
        BMUs[kk] = BMU

    
    # lets put together all the data for each time step and calculate the score!!!
    crps_som = np.empty((len(dist),BMUs.shape[0]))  # (nodes, forecasts)
    crps_som.fill(np.nan)
    crps_clim = np.empty((len(dist),BMUs.shape[0]))
    crps_clim.fill(np.nan)
    crps_wind = np.empty((len(dist),BMUs.shape[0]))
    crps_wind.fill(np.nan)
    obs_node = np.empty((len(dist),BMUs.shape[0]))
    obs_node.fill(np.nan)

    for i in range(BMUs.shape[0]):
        BMU = BMUs[i]
        dist_forecast = dist[BMU]
        date_init = z_forecast.isel(time=i).time.values
        date_forecast = date_init + t
        month = date_forecast.astype('datetime64[M]').astype(int) % 12 + 1
        day = (date_forecast.astype('datetime64[D]') - date_forecast.astype('datetime64[M]')).astype(int) + 1
        #cl = climo.sel(day=day,month=month).nighttime.values  # deterministic climo - don't need right now
        wind = wind_forecast.sel(time=date_init).wind.values
        ob = obs_test.to_xarray().sel(index=date_forecast).Wind.values
        obs_node[BMU,i] = ob
        crps_som[BMU,i] =(ps.crps_ensemble(ob, dist_forecast))
        crps_clim[BMU,i] =(ps.crps_ensemble(ob, obs_train))
        crps_wind[BMU,i] =(ps.crps_ensemble(ob, wind))

    for i in range(crps_som.shape[0]):
        
        print(np.nanmean(1-crps_som[i,:]/crps_clim[i,:]))

    print('done')


    