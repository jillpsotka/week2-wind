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


def era5_prep(ds, period=slice("2009-10-01","2023-05-01")):

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
        # to remember: when xarray resamples, it names the interval by its starting point

    return ds


def weighted_mean(obj):
    # map to take a weighted mean with first and last points weighed less
    weights = np.ones(len(obj.step))
    weights[0] = 0.5
    weights[-1] = 0.5
    weights = weights / weights.sum()
    weights = xr.DataArray(weights,dims='step',coords={'step':obj.step.values})
    return obj.weighted(weights).mean(dim='step')
def weighted_mean_rolling(obj,axis=4):
    # map to take a weighted mean with first and last points weighed less
    weights = np.ones(obj.shape[-1])
    weights[0] = 0.5
    weights[-1] = 0.5
    weights = weights / weights.sum()

    return np.sum(obj * weights, axis=axis)
def findMiddle(input_list,axis=2):
    middle = float((input_list.shape[-1]))/2
    if middle < 1:
        return input_list[:,:,0]
    if middle % 2 != 0:
        return input_list[:,:,int(middle)]
    else:
        return np.mean((input_list[:,:,int(middle)], input_list[:,:,int(middle-1)]),axis=0)


def resample_mean(ds, type_str,res=6):
    # resample 5-minute obs data
    # res is number of hours per resampling period
    #origin = int((res / 2) % 24)  # modulus makes it work when res is >24
    if res == 6 or res == 12:
        origin = 0  # 10am - 4pm PST for 6-h
    else:
        origin = 0  # 7pm - 7am PST

    res_str = str(res) + 'H'
    if type_str == 'obs':  # separate this cuz obs are 5-minute and era is 1-hr
        nan_indices=np.argwhere(np.isnan(ds.rolling(index=int(res*12)+1,min_periods=int(res*12/2),center=True).mean().Wind.values)).squeeze()  # this will tag any periods that are >1/2 nan
        ds.Wind[nan_indices] = np.nan
        ds = ds.resample(index=res_str,origin=datetime(2007,12,1,origin)).mean()
    elif type_str == 'era':
        ds = ds.resample(time=res_str,origin=datetime(2007,12,1,origin)).mean()
    elif type_str == 'gefs':
        ds10 = ds.sel(step=slice(np.array(int(9.75*24*1e9*60*60),dtype='timedelta64[ns]'),  # after day 10
                                np.array(int(16*24*1e9*60*60),dtype='timedelta64[ns]')))
        ds = ds.sel(step=slice(np.array(int(5*24*1e9*60*60),dtype='timedelta64[ns]'),  # up to day 10
                                np.array(int(11*24*1e9*60*60),dtype='timedelta64[ns]')))
        
        ds['step'] = ds.step.values.astype('datetime64[ns]')
        ds10['step'] = ds10.step.values.astype('datetime64[ns]')

        # using a custom map mean function
        weight = xr.DataArray([0.25, 0.5, 0.25], dims=['window'])
        ds = ds.gh.rolling(step=3,center=True).construct('window').dot(weight)  # take weighted rolling averages
        ds = ds.dropna(dim='step',how='all')

        weight = xr.DataArray([0.5, 0.5], dims=['window'])
        ds10 = ds10.gh.rolling(step=2,center=True).construct('window').dot(weight)  # take weighted rolling averages
        ds10 = ds10.dropna(dim='step',how='all')

        ds = xr.concat([ds, ds10],dim='step').drop_duplicates(dim='step')

        ds = ds.resample(step='6H',origin=datetime(2007,12,1,origin)).last()
        ds['step'] = ds.step.values.astype('timedelta64[ns]')
        #ds['time'] = ds.time.values + np.timedelta64(origin,'h')  # time stuff is weird

    elif type_str == 'gefs-wind':
        if res == 168:
            ds['step'] = ds.step.values.astype('datetime64[ns]')
            ds = ds.wind.rolling(step=54,center=True,min_periods=40).mean()
            ds = ds.dropna(dim='step',how='all')
            ds = ds.resample(step=res_str).reduce(findMiddle)
            ds['step'] = ds.step.values.astype('timedelta64[ns]')
            return ds
        # i think split things into before day 10 and after to simplify rolling stuff?
        ds10 = ds.sel(step=slice(np.array(int(9.75*24*1e9*60*60),dtype='timedelta64[ns]'),  # after day 10
                                np.array(int(16*24*1e9*60*60),dtype='timedelta64[ns]')))
        ds = ds.sel(step=slice(np.array(int(5*24*1e9*60*60),dtype='timedelta64[ns]'),  # up to day 10
                                np.array(int(11*24*1e9*60*60),dtype='timedelta64[ns]')))
        weights = np.ones(int((res/3)+1))
        weights[0] = 0.5
        weights[-1] = 0.5
        weights = weights / weights.sum()
        ds['step'] = ds.step.values.astype('datetime64[ns]')
        # using a custom map mean function
        weight = xr.DataArray(weights, dims=['window'])
        ds = ds.wind.rolling(step=len(weight),center=True,min_periods=int(res/(3*2))).construct('window').dot(weight)  # take weighted rolling averages
        ds = ds.dropna(dim='step',how='all')

        weights = np.ones(int((res/6)+1))
        weights[0] = 0.5
        weights[-1] = 0.5
        weights = weights / weights.sum()
        ds10['step'] = ds10.step.values.astype('datetime64[ns]')
        # using a custom map mean function
        weight = xr.DataArray(weights, dims=['window'])
        ds10 = ds10.wind.rolling(step=len(weight),center=True,min_periods=res/(6*2)).construct('window').dot(weight)  # take weighted rolling averages
        ds10 = ds10.dropna(dim='step',how='all')

        # merge before resampling
        ds = xr.concat([ds, ds10],dim='step').drop_duplicates(dim='step')
        ds = ds.resample(step=res_str,origin=datetime(2007,12,1,origin)).reduce(findMiddle)#last()
        ds['step'] = ds.step.values.astype('timedelta64[ns]')
        #ds['time'] = ds.time.values + np.timedelta64(origin,'h')  # time stuff is weird
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
    obs = obs.sel(index=times_new.values)  # get rid of obs that aren't in gefs


    return obs, gefs



if __name__ == "__main__":
    #dates = slice("2012-11-20", "2022-12-31")

    #obs = cleaned_obs(dates)
    #obs = xr.open_dataset('data/bm_cleaned_all.nc').sel(index='2015')

    #gefs = xr.open_dataset('data/gh-reanalysis-2014-01.nc').sel(isobaricInhPa=500)
    #gefs = gefs_reanalysis(gefs, period=dates, res=6)
    era1 = xr.open_dataset('era-850-2009.grib')
    era1 = era5_prep(era1)
    era1 = era1.drop_vars(['valid_time','number','step'])
    era1 = era1.rename({'isobaricInhPa':'level'})
    era = xr.open_dataset('era-850-2009-2022.nc')
    era = xr.concat([era1,era],dim='time')
    era = era.drop_duplicates('time')
    era.to_netcdf('era-850-2009-2022.nc')
    #low_pass_filter(obs,'obs',12)
    #era = era5_prep(era)
    #era = resample_mean(era)
    #era = era5(era, dates)

    #dates_obs_gefs(obs, gefs)
    print("done")



