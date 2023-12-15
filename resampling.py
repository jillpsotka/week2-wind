import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xarray as xr


def gefs_reanalysis(ds, period=slice("2012-11-20", "2019-12-25"), origin=0, res=6):
    res_str = str(res) + 'H'

    ds = ds.sel(time=period)
    ds = ds.resample(time=res_str).mean()
    # e.g. 12-hour resampling will make 12:00 be an avg of 12:00 and 18:00.

    return ds



def cleaned_obs(period=slice("2012-11-20", "2019-12-25"), res=6):
    # right now this is set up for 6-hourly
    # res is number of hours per resampling period
    # gefs reanalysis comes at 00, 06, 12, 18
    # for now, want obs to be surrounding averages. so make the origin offset from 00:
    origin = int(res / 2)

    obs = xr.open_dataset('data/bm_cleaned.nc')
    obs = obs.sel(index=period)
    ds1 = obs.to_dataframe()

    # resampling at origin (UTC)
    res_str = str(res) + 'H'
    ds = ds1.groupby(pd.Grouper(freq=res_str,origin=datetime(2011,12,1,origin))).agg(['mean','count']).swaplevel(0,1,axis=1)
    ds = ds['mean'].where(ds['count']>=(12*res/2))  # only keep periods that have >1/2 of obs
    # for example, 12:00 point for 6 hourly will have avg from 12:00 to 17:55

    ds.index += pd.DateOffset(hours=3)
    # shift the time axis so that a 12:00 point for 6 hourly will have avg from 9:00-14:55
    # this way the time axis matches the gefs times

    print(np.isnan(ds.Wind.values).sum(), 'missing', res_str, 'periods out of', len(ds.Wind.values))
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
    dates = slice("2012-11-20", "2019-12-25")

    obs = cleaned_obs(dates)

    gefs = xr.open_dataset('data/gh-reanalysis-2014-01.nc').sel(isobaricInhPa=500)
    gefs = gefs_reanalysis(gefs, period=dates, res=6)

    dates_obs_gefs(obs, gefs)



