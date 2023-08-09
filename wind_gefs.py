import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd
from scipy import interpolate


def look_at_obs(period, res):
    ds = xr.open_dataset('data/bm_12.nc')
    ds = ds.sel(index=period).resample(index=res,skipna=False).mean()

    nans = np.isnan(ds.Wind.values)
    if True in nans:
        print('nan found')
    return ds


def look_at_gefs(period, res):
    ds = xr.open_dataset('data/wind-2018-12-26-2019-12-31-0.nc')
    ds = ds.sel(time=period)
    ds = ds.resample(step=res,skipna=True).mean()  # taking averages
    return ds


def split_days(arr, n, gefs):
    # n is the day, but we have indexing that starts at day 6...
    n = (n-6)*2
    d = np.empty((int(2*arr[n,:].size)))
    d[0::2] = arr[n,:]
    d[1::2] = arr[n+1,:]

    valid_time = np.empty_like(d,dtype='datetime64[ns]')
    valid_time[0::2] = gefs.time + gefs.step[n]
    valid_time[1::2] = gefs.time + gefs.step[n+1]

    dr = xr.Dataset(data_vars=dict(d6=(["index"],d)),coords=dict(index=valid_time))

    return dr


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
    
    w10 = interpolate.griddata(coords, vals10, loc, method='linear').squeeze()*3.6  
    w100 = interpolate.griddata(coords, vals100, loc, method='linear').squeeze()*3.6

    # vertical interp using power law
    alpha = np.log(w100/w10) / np.log(100/10)  # shear exponent
    w80 = w100*(80/100)**alpha  # array (step,times)

    return w80


def combine_obs_gefs(obs, gefs):
    arr = interp(gefs)
    # arr is the interpolated forecast of shape (step, times)
    d6 = split_days(arr, 6, gefs)
    d7 = split_days(arr, 7, gefs).rename_vars(dict(d6='d7'))
    d8 = split_days(arr, 8, gefs).rename_vars(dict(d6='d8'))
    d9 = split_days(arr, 9, gefs).rename_vars(dict(d6='d9'))
    d10 = split_days(arr, 10, gefs).rename_vars(dict(d6='d10'))
    d11 = split_days(arr, 11, gefs).rename_vars(dict(d6='d11'))
    d12 = split_days(arr, 12, gefs).rename_vars(dict(d6='d12'))
    d13 = split_days(arr, 13, gefs).rename_vars(dict(d6='d13'))
    d14 = split_days(arr, 14, gefs).rename_vars(dict(d6='d14'))

    merged = xr.merge((obs,d6,d7,d8,d9,d10,d11,d12,d13,d14))

    return merged


def scatter_plot(d):
    plt.plot(d.Wind,d.Wind,c='black')
    plt.scatter(d.Wind, d.d6, label='day 6')
    plt.scatter(d.Wind, d.d14, label='day 14')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print('starting', datetime.now())
    res = '12H'  # resolution to avg
    dates_obs = pd.date_range('2019-10-01','2019-10-16',freq=res)
    dates_gefs = pd.date_range('2019-09-26','2019-10-13')

    obs = look_at_obs(dates_obs, res)
    gefs = look_at_gefs(dates_gefs, res)
    ds = combine_obs_gefs(obs, gefs)
    scatter_plot(ds)
    
    
    print('done',datetime.now())