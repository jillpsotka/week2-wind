from herbie import FastHerbie
import numpy as np
import pandas as pd
import xarray as xr
import time
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from toolbox import EasyMap, pc
from datetime import datetime


def get_z():
    # Herbie : quick
    tic = time.perf_counter()
    H = FastHerbie(period, model="gefs_reforecast",fxx=[100,270], member=0, variable_level="hgt_pres_abv700mb")
    toc = time.perf_counter()
    print(f"finished FastHerbie in {toc - tic:0.4f} seconds. Now putting into xarray...")

    # xarray: takes ~15 minutes per month, look into cutting more?
    ds = H.xarray(":500 mb:")
    toc2 = time.perf_counter()
    print(f"finished xarray in {(toc2 - toc)/60:0.4f} minutes. Now clipping and saving...")

    steps = np.arange(168,360,12,dtype='timedelta64[ns]')*(1000000000*60*60)  # day 7-14 every 12 hours
    lats = np.arange(45,65.5,0.5)[::-1]
    lons = np.arange(230,250.5,0.5)
    ds = ds.sel(latitude=lats,longitude=lons,step=steps)
    ds.gh.to_netcdf('data/test.nc')
    toc3 = time.perf_counter()
    print(f"finished saving in {(toc3 - toc2):0.4f} seconds.")

    print('done')


def get_wind():
    # Herbie : quick
    tic = time.perf_counter()
    H = FastHerbie(period, model="gefs_reforecast",fxx=[100,270], member=0, variable_level="ugrd_hgt")
    H2 = FastHerbie(period, model="gefs_reforecast",fxx=[100,270], member=0, variable_level="vgrd_hgt")

    toc = time.perf_counter()
    print(f"finished FastHerbie in {toc - tic:0.4f} seconds. Now putting into xarray...")

    # xarray: takes ~15 minutes per month, look into cutting more?
    u_10 = H.xarray(":UGRD:10 m")
    u_100 = H.xarray(":UGRD:100 m")
    print('Halfway there..', datetime.now())
    v_10 = H2.xarray(":VGRD:10 m")
    v_100 = H2.xarray(":VGRD:100 m")
    toc2 = time.perf_counter()
    print(f"finished xarray in {(toc2 - toc)/60:0.4f} minutes. Now clipping and saving...")

    steps = np.arange(144,360,12,dtype='timedelta64[ns]')*(1000000000*60*60)  # day 7-14 every 12 hours
    lats = np.arange(55.5,56.5,0.5)[::-1]
    lons = np.arange(240,241,0.5)
    u_10 = u_10.sel(latitude=lats,longitude=lons,step=steps)
    #u_100 = u_100.sel(latitude=lats,longitude=lons,step=steps)
    v_10 = v_10.sel(latitude=lats,longitude=lons,step=steps)
    #v_100 = v_100.sel(latitude=lats,longitude=lons,step=steps)

    u_10 = u_10.assign(wind=np.sqrt(u_10.u10**2 + v_10.v10**2))
    u_100.assign(wind=np.sqrt(u_100.u100**2 + v_100.v100**2))

    u_10.wind.to_netcdf('data/test_wind10.nc')
    u_100.wind.to_netcdf('data/test_wind100.nc')

    toc3 = time.perf_counter()
    print(f"finished saving in {(toc3 - toc2):0.4f} seconds.")

    print('done')


if __name__ == '__main__':
    print("starting..", datetime.now())

    # set search params
    period = pd.date_range(
    start="2012-11-25",
    end="2012-12-25")

    get_wind()

    print(datetime.now())