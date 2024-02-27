import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import xarray as xr


def gefs_reanalysis(ds, period=slice("2012-11-20", "2019-12-25"), origin=0, res=6):
    res_str = str(res) + 'H'

    ds = ds.sel(time=period)
    ds = ds.resample(time=res_str).mean()
    # e.g. 12-hour resampling will make 12:00 be an avg of 12:00 and 18:00.

    return ds


def era5_prep(ds, period=slice("2012-11-20","2022-01-01")):

    ds = ds.sel(time=period)
    ds = ds[{'longitude':slice(None,None,2),'latitude':slice(None,None,2)}]
    ds['z'] = ds['z']/9.80665
    ds = ds.rename_vars({'z':'gh'})

    return ds


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y



def low_pass_filter(ds,window_size=3):
    # do low pass filter
    # w_max = int((window_size - 1) / 2)
    # w_arr = np.concatenate((np.arange(0, w_max), np.arange(0, w_max)[:-1:-1]))[1:-2]  # make smoothed weights
    # w_arr = w_arr / np.sum(w_arr)  # normalize weights
    # weight = xr.DataArray(w_arr, dims=['window'])
    # ds = ds.gh.rolling(time=window_size,min_periods=int(window_size/2)).construct('window').dot(weight).shift(time=-window_size)
    # resolution will stay at whatever it was originally but cross-referencing with obs will solve that?
    # Filter requirements.
    order = 6
    fs = (1/5)/60       # sample rate, Hz (1/s)    (5-min)
    cutoff = (1/60)/60  # desired cutoff frequency of the filter, Hz   (1-hr)

    # Get the filter coefficients so we can check its frequency response.
    y = butter_lowpass(ds,cutoff, fs, order)
    
    return ds


def resample_mean(ds1, type_str,res=12):
    # resample 5-minute obs data
    # res is number of hours per resampling period
    #origin = int((res / 2) % 24)  # modulus makes it work when res is >24
    if res == 6 or res == 8:
        origin = 23  # 9am-3pm PST for 6-h, or 7am-3pm, 3pm-11pm, 11pm-7am PST for 8-h
    else:
        origin = 2  # 6pm - 6am PST
    ds1 = ds1.to_dataframe()

    res_str = str(res) + 'H'
    ds = ds1.groupby(pd.Grouper(freq=res_str,origin=datetime(2011,12,1,origin))).agg(['mean','count']).swaplevel(0,1,axis=1)
    if type_str == 'obs':  # separate this cuz obs are 5-minute and era is 1-hr
        ds = ds['mean'].where(ds['count']>=(12*res/2))  # only keep periods that have >1/2 of obs
    elif type_str == 'era':
        ds = ds['mean'].where(ds['count']>=(res/2))
    elif type_str == 'gefs':
        ds = ds['mean'].where(ds['count']>=(res/6))
    # for example, 12:00 point for 6 hourly will have avg from 12:00 to 17:55
    
    #ds.index += pd.DateOffset(hours=int(res / 2))
    # shift the time axis so that a 12:00 point for 6 hourly will have avg from 9:00-14:55

    return ds



def dates_obs_gefs(obs, gefs):
    # make times of obs and gefs line up
    obs = obs.to_xarray()
 
    obs = obs.where(~np.isnan(obs.Wind),drop=True)  # get rid of nan obs
    gefs = gefs.where(~np.isnan(gefs.gh),drop=True)  # get rid of nan gefs

    times = gefs.time # list of gefs times
    indices = times.isin(obs.index)  # indices of valid times that have obs

    times_new = times.where(indices, drop=True)  # valid times that have obs

    gefs = gefs.sel(time = times_new)  # get rid of gefs times that don't have obs
    obs = obs.sel(index=times_new)#.Wind.to_numpy()  # get rid of obs that aren't in gefs


    return obs, gefs



if __name__ == "__main__":
    dates = slice("2012-11-20", "2022-12-31")

    #obs = cleaned_obs(dates)

    #gefs = xr.open_dataset('data/gh-reanalysis-2014-01.nc').sel(isobaricInhPa=500)
    #gefs = gefs_reanalysis(gefs, period=dates, res=6)
    era = xr.open_dataset('era-reanalysis-2012-2022-1h.nc')
    era = resample_mean(era)
    #era = era5(era, dates)

    #dates_obs_gefs(obs, gefs)
    print("done")



