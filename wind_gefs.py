import numpy as np
import xarray as xr
from datetime import datetime,timedelta
import pandas as pd
from scipy import interpolate


# def load_obs(period, res):
#     ds = xr.open_dataset('data/bm_12.nc')
#     ds = ds.sel(index=period).resample(index=res,skipna=False).mean()
#     return ds


def load_obs(period, res=12):
    # res is number of hours per resampling period
    obs = xr.open_dataset('data/bm_cleaned.nc')
    obs = obs.sel(index=period)
    ds1 = obs.to_dataframe()

    # resampling at origin which is 3am UTC (7pm PST)
    res_str = str(res) + 'H'
    ds = ds1.groupby(pd.Grouper(freq=res_str,origin=datetime(2011,12,1,3))).agg(['mean','count']).swaplevel(0,1,axis=1)
    ds = ds['mean'].where(ds['count']>=(12*res/2))  # only keep periods that have >1/2 of obs
    # for example, 12:00 point for 6 hourly will have avg from 12:00 to 17:55

    #times = pd.date_range(datetime(2012,12,1,12),datetime(2012,12,1,18),freq='5T')
    ds = ds.shift(periods=1)
    # now 12:00 point for 6 hourly will have avg from 6:00-11:55 (this matches processed GEFS)

    print(np.isnan(ds.Wind.values).sum(), 'missing', res_str, 'periods out of', len(ds.Wind.values))
    return ds


def load_gefs(period, res=12):
    # todo: only really set up for 12 hourly
    ds0 = xr.open_dataset('data/wind-2018-12-26-2019-12-31-0.nc')
    ds1 = xr.open_dataset('data/wind-2018-12-22-2019-12-27-1.nc')
    ds2 = xr.open_dataset('data/wind-2018-12-22-2019-12-27-2.nc')
    ds3 = xr.open_dataset('data/wind-2018-12-22-2019-12-27-3.nc')
    ds4 = xr.open_dataset('data/wind-2018-12-22-2019-12-27-4.nc')

    ds = xr.concat((ds0, ds1, ds2, ds3, ds4),dim='member')
    ds = ds.sel(time=period)
    return ds


def resample_gefs(ds):
    res_str = str(res) + 'H'

    # taking rolling means that include the start and end point of the wind.
    # 15:00 in rolling corresponds to 03:00-15:00
    ind_159 = np.where(ds.step.values==np.timedelta64(int(159*1e9*3600),'ns'))[0][0]  # index of 159H
    ind_252 = np.where(ds.step.values==np.timedelta64(int(252*1e9*3600),'ns'))[0][0]
    #rolling_1 = ds.rolling(step=5).mean().sel(step=pd.timedelta_range("159H", periods=8,freq="12H"))  # changes past day 10
    rolling_1 = ds.rolling(step=5).mean().isel(step=slice(ind_159,ind_252,4))
    #rolling_2 = ds.rolling(step=2).mean().sel(step=pd.timedelta_range("252H", periods=10,freq="12H"))
    rolling_2 = ds.rolling(step=2).mean().isel(step=slice(ind_252,200,2))
    rolling = xr.concat([rolling_1, rolling_2],dim='step')

    return rolling


def interp(gefs):
    # horizontal interpolation
    coords = np.array([[gefs.latitude.values[0],gefs.longitude.values[0]],  # (56,239.5)
                       [gefs.latitude.values[0],gefs.longitude.values[1]],
                       [gefs.latitude.values[1],gefs.longitude.values[0]],
                       [gefs.latitude.values[1],gefs.longitude.values[1]]])
    loc = np.array([55.6986, 360-120.4306])  # insert lat/lon of wind farm
    vals10 = np.array([gefs.wind10.isel(latitude=0,longitude=0),
                       gefs.wind10.isel(latitude=0,longitude=1),
                       gefs.wind10.isel(latitude=1,longitude=0),
                       gefs.wind10.isel(latitude=1,longitude=1)])
    vals100 = np.array([gefs.wind100.isel(latitude=0,longitude=0),
                       gefs.wind100.isel(latitude=0,longitude=1),
                       gefs.wind100.isel(latitude=1,longitude=0),
                       gefs.wind100.isel(latitude=1,longitude=1)])
    
    w10 = interpolate.griddata(coords, vals10, loc, method='linear').squeeze()  
    w100 = interpolate.griddata(coords, vals100, loc, method='linear').squeeze()

    # vertical interp using power law
    alpha = np.log(w100/w10) / np.log(100/10)  # shear exponent
    w80 = w100*(80/100)**alpha  # array (step,members,times)

    return w80


def split_days(arr, n, gefs):
    # n is the day, but we have indexing that starts at day 6...
    n = (n-6)*2
    d = np.empty((arr[n,:,:].shape[0],2*arr[n,:,:].shape[1]))
    d[:,0::2] = arr[n,:,:]  # filling every other value with the 00:00 forecast for day n
    d[:,1::2] = arr[n+1,:,:]  # filling every other value with the 12:00 forecast for day n

    valid_time = np.empty_like(d[0,:],dtype='datetime64[ns]')
    valid_time[0::2] = gefs.time + gefs.step[n]
    valid_time[1::2] = gefs.time + gefs.step[n+1]

    dr = xr.Dataset(data_vars=dict(d6=(["member","index"],d)),coords=dict(index=valid_time,member=np.arange(5)))

    return dr


def combine_obs_gefs(obs, gefs):
    arr = interp(gefs)
    gefs = gefs.assign(wind=(['member','time','step'],arr)).drop_dims(['longitude','latitude'])
    gefs = gefs.drop_vars(['valid_time','number'])

    # arr is the interpolated forecast of shape (step, times)
    # d6 = split_days(arr, 6, gefs)  # this adds back in time dimension
    # d7 = split_days(arr, 7, gefs).rename_vars(dict(d6='d7'))
    # d8 = split_days(arr, 8, gefs).rename_vars(dict(d6='d8'))
    # d9 = split_days(arr, 9, gefs).rename_vars(dict(d6='d9'))
    # d10 = split_days(arr, 10, gefs).rename_vars(dict(d6='d10'))
    # d11 = split_days(arr, 11, gefs).rename_vars(dict(d6='d11'))
    # d12 = split_days(arr, 12, gefs).rename_vars(dict(d6='d12'))
    # d13 = split_days(arr, 13, gefs).rename_vars(dict(d6='d13'))
    # d14 = split_days(arr, 14, gefs).rename_vars(dict(d6='d14'))

    # merged = xr.merge((obs,d6,d7,d8,d9,d10,d11,d12,d13,d14))
    #merged = merged.expand_dims(dim={"member":np.arange(5)})  # all member identical numbers right now
    #merged['Wind'] = merged['Wind'].sel(member=0).drop('member')

    return gefs


def calculate_climo(obs, ds):

    # months = obs['index.month']
    # days = obs['index.day']

    climo = xr.open_dataset('data/climo-12h.nc')
    # climo = climo.sel(level_0=months, level_1=days)

    # daily_climo = climo['Wind'].rename('climo')
    # ds['climo'] = daily_climo

    return climo



if __name__ == '__main__':
    print('starting', datetime.now())
    res = 12  # resolution to avg
    dates = slice("2012-11-20", "2019-12-25")
    gefs = xr.open_dataset('/Users/jpsotka/Nextcloud/geo-height-data/gh-2012-11-20-2017-12-25-4.nc')

    #obs = load_obs(dates, res)
    #gefs = load_gefs(dates, res)
    gefs = resample_gefs(gefs)
    #gefs = combine_obs_gefs(obs, gefs)
    #climo = calculate_climo(obs, gefs)

    gefs.to_netcdf('data/geo-height-2012-2017-12h-4.nc')
    
    print('done',datetime.now())