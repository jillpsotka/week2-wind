import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import signal
import matplotlib.pyplot as plt
import xarray as xr
import sys


def gefs_reanalysis(ds, period=slice("2012-11-20", "2019-12-25"), origin=0, res=6):
    res_str = str(res) + 'H'

    ds = ds.sel(time=period)
    ds = ds.resample(time=res_str).mean()
    # e.g. 12-hour resampling will make 12:00 be an avg of 12:00 and 18:00.

    return ds


def era5_prep(ds, period=slice("2009-10-01","2022-01-01")):

    ds = ds.sel(time=period)
    ds = ds[{'longitude':slice(None,None,2),'latitude':slice(None,None,2)}]
    ds['z'] = ds['z']/9.80665
    ds = ds.rename_vars({'z':'gh'})

    return ds


def low_pass_filter(ds,type_str,res):
    # used https://www.earthinversion.com/techniques/Time-Series-Analysis-Filtering-or-smoothing-data/ 

    # from the looks of it, overarching gh patterns are 3-4 days long, whereas wind are ~12 hours long.

    res_str = str(res) + 'H'
    if res == 6 or res == 8:
        origin = 23  # 9am-3pm PST for 6-h, or 7am-3pm, 3pm-11pm, 11pm-7am PST for 8-h
    else:
        origin = 2  # 6pm - 6am PST

    if type_str == 'era' or type_str == 'gefs':
        fs = (1/60)/60       # sample rate, Hz (1/s)    (1 hr)
        cutoff=0.03          # fraction of nyquist frequency, smaller fraction will do more, 0.02 turns gh data into ~3-4 day patterns

        b, a = signal.butter(5, cutoff, btype='lowpass')
        #ds = ds.interpolate_na(dim='time',method='linear')
        dUfilt = signal.filtfilt(b, a, ds.gh.values,axis=0)  # axis has to be time axis
        ds.gh.values = dUfilt
        ds = ds.resample(time=res_str,origin=datetime(2007,12,1,origin)).mean()  # resample and take mean

    elif type_str == 'obs':
        # first deal with the fact that obs have nan values which mess up the filter
        # interplote the nans but save their indices to discard afterwards
        nan_indices=np.argwhere(np.isnan(ds.rolling(index=int(res*12)+1,min_periods=int(res*12/2),center=True).mean().Wind.values)).squeeze()  # this will tag any periods that are >1/2 nan
        ds.Wind[nan_indices] = np.nan
        ds = ds.interpolate_na(dim='index',method='linear',fill_value=5)

        fs = (1/5)/60       # sample rate, Hz (1/s)    (5 min)
        cutoff=0.02        # 0.02 ~6hrs, 0.01 ~ 12hrs, 0.003 ~1day
        b, a = signal.butter(5, cutoff, btype='lowpass')

        dUfilt = signal.filtfilt(b, a, ds.Wind.values,axis=0)  # axis has to be time axis

        dUfilt[nan_indices] = np.nan  # get rid of nan again
        ds.Wind.values = dUfilt
        ds = ds.resample(index=res_str,origin=datetime(2007,12,1,origin)).mean()  # resample and take mean

    return ds


def resample_mean(ds1, type_str,res=12):
    # resample 5-minute obs data
    # res is number of hours per resampling period
    #origin = int((res / 2) % 24)  # modulus makes it work when res is >24
    if res == 6 or res == 8:
        origin = 23  # 9am-3pm PST for 6-h, or 7am-3pm, 3pm-11pm, 11pm-7am PST for 8-h
    else:
        origin = 2  # 6pm - 6am PST

    res_str = str(res) + 'H'
    if type_str == 'obs':  # separate this cuz obs are 5-minute and era is 1-hr
        ds1 = ds1.to_dataframe()
        ds = ds1.groupby(pd.Grouper(freq=res_str,origin=datetime(2007,12,1,origin))).agg(['mean','count']).swaplevel(0,1,axis=1)
        ds = ds['mean'].where(ds['count']>=(12*res/2))  # only keep periods that have >1/2 of obs
        ds = ds.to_xarray()
    elif type_str == 'era' or type_str == 'gefs':
        ds = ds1.resample(time=res_str,origin=datetime(2011,12,1,origin)).mean()
    else:
        print('type string must be one of obs/era/gefs. exiting')
        sys.exit()
    # for example, 12:00 point for 6 hourly will have avg from 12:00 to 17:55
    
    #ds.index += pd.DateOffset(hours=int(res / 2))
    # shift the time axis so that a 12:00 point for 6 hourly will have avg from 9:00-14:55

    return ds



def dates_obs_gefs(obs, gefs):
    # make times of obs and gefs line up
 
    obs = obs.where(~np.isnan(obs.Wind),drop=True)  # get rid of nan obs
    gefs = gefs.where(~np.isnan(gefs.gh),drop=True)  # get rid of nan gefs

    times = gefs.time # list of gefs times
    indices = times.isin(obs.index)  # indices of valid times that have obs

    times_new = times.where(indices, drop=True)  # valid times that have obs

    gefs = gefs.sel(time = times_new)  # get rid of gefs times that don't have obs
    obs = obs.sel(index=times_new)  # get rid of obs that aren't in gefs


    return obs, gefs



if __name__ == "__main__":
    #dates = slice("2012-11-20", "2022-12-31")

    #obs = cleaned_obs(dates)
    #obs = xr.open_dataset('data/bm_cleaned_all.nc').sel(index='2015')

    #gefs = xr.open_dataset('data/gh-reanalysis-2014-01.nc').sel(isobaricInhPa=500)
    #gefs = gefs_reanalysis(gefs, period=dates, res=6)
    era7 = xr.open_dataset('era-2009-2021-700.nc')
    #low_pass_filter(obs,'obs',12)
    #era = era5_prep(era)
    #era = resample_mean(era)
    #era = era5(era, dates)

    #dates_obs_gefs(obs, gefs)
    print("done")



