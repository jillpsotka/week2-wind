import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd


def look_at_obs(period):
    ds = xr.open_dataset('data/bm_12.nc')
    ds = ds.sel(index=period)

    nans = np.isnan(ds.Wind.values)
    if True in nans:
        print('nan found')
    return ds


def look_at_gefs(period):
    ds = xr.open_dataset('data/wind-2018-12-26-2019-12-31')
    ds = ds.sel(time=period)
    return ds


def forecast(gefs):
    # horizontal interpolation
    coords = np.array([[gefs.latitude.values[0],gefs.longitude.values[0]],
                       [gefs.latitude.values[0],gefs.longitude.values[1]],
                       [gefs.latitude.values[1],gefs.longitude.values[0]],
                       [gefs.latitude.values[1],gefs.longitude.values[1]]])
    loc = np.array([55.6986, 360-120.4306])  # insert lat/lon of wind farm

    print('hi')


if __name__ == '__main__':
    print('starting', datetime.now())
    dates_obs = pd.date_range('2019-01-01','2020-01-01',freq='12H')
    #obs = look_at_obs(dates_obs)
    dates_gefs = pd.date_range('2018-12-26','2019-01-26')
    gefs = look_at_gefs(dates_gefs)
    forecast(gefs)
    
    print('done',datetime.now())