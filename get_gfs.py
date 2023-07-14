from herbie import FastHerbie
import numpy as np
import pandas as pd
import xarray as xr
import time
import cartopy.crs as ccrs
import glob
from toolbox import EasyMap, pc
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor


def make_dates(start_date="2012-11-25" ,number=1, max=30):
    # make __ pandas date ranges with max size 30 days
    dates = []
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    for i in range(number):
        dates.append(pd.date_range(start=(start_date+timedelta(days=i*max)),periods=max))

    return dates


def get_z(period):
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


def get_wind_10(n, period):
    # Herbie : quick
    tic = time.perf_counter()
    H = FastHerbie(period, model="gefs_reforecast",fxx=[100,270], member=0, variable_level="ugrd_hgt")
    H2 = FastHerbie(period, model="gefs_reforecast",fxx=[100,270], member=0, variable_level="vgrd_hgt")

    toc = time.perf_counter()
    print(f"finished FastHerbie in {toc - tic:0.4f} seconds. Now putting into xarray...")

    # xarray: takes a long time :(
    steps = np.arange(144,360,12,dtype='timedelta64[ns]')*(1000000000*60*60)  # day 6-14 every 12 hours
    lats = np.arange(55.5,56.5,0.5)[::-1]
    lons = np.arange(240,241,0.5)

    u_10 = H.xarray(":UGRD:10 m")
    u_10 = u_10.sel(latitude=lats,longitude=lons,step=steps)
    u_10 = u_10.drop('heightAboveGround')  # this coordinate gets in the way
    
    print('Halfway there..', datetime.now())
    v_10 = H2.xarray(":VGRD:10 m")
    v_10 = v_10.drop('heightAboveGround')  # this coordinate gets in the way
    v_10 = v_10.sel(latitude=lats,longitude=lons,step=steps)

    toc2 = time.perf_counter()
    print(f"finished xarray in {(toc2 - toc)/60:0.4f} minutes. Now saving...")

    w10 = u_10.assign(wind=np.sqrt(u_10.u10**2 + v_10.v10**2))
    title = 'data/' + str(n) + 'wind10.nc'
    w10.wind.to_netcdf(title)

    toc3 = time.perf_counter()
    print(f"finished saving in {(toc3 - toc2):0.4f} seconds.")


def get_wind_100(n, period):
    # Herbie : quick
    tic = time.perf_counter()
    H = FastHerbie(period, model="gefs_reforecast",fxx=[100,270], member=0, variable_level="ugrd_hgt")
    H2 = FastHerbie(period, model="gefs_reforecast",fxx=[100,270], member=0, variable_level="vgrd_hgt")

    toc = time.perf_counter()
    print(f"finished FastHerbie in {toc - tic:0.4f} seconds. Now putting into xarray...")

    # xarray: takes a long time :(
    steps = np.arange(144,360,12,dtype='timedelta64[ns]')*(1000000000*60*60)  # day 6-14 every 12 hours
    lats = np.arange(55.5,56.5,0.5)[::-1]
    lons = np.arange(240,241,0.5)

    u_100 = H.xarray(":UGRD:100 m")
    u_100 = u_100.sel(latitude=lats,longitude=lons,step=steps)
    u_100 = u_100.drop('heightAboveGround')
    
    print('Halfway there..', datetime.now())

    v_100 = H2.xarray(":VGRD:100 m")
    v_100 = v_100.drop('heightAboveGround')
    v_100 = v_100.sel(latitude=lats,longitude=lons,step=steps)
    toc2 = time.perf_counter()
    print(f"finished xarray in {(toc2 - toc)/60:0.4f} minutes. Now saving...")

    w100 = u_100.assign(wind=np.sqrt(u_100.u100**2 + v_100.v100**2))
    title = 'data/' + str(n) + 'wind100.nc'
    w100.wind.to_netcdf(title)

    toc3 = time.perf_counter()
    print(f"finished saving in {(toc3 - toc2):0.4f} seconds.")


def merge_data():
    #take multiple ncs of gefs and combine them
    file_list = glob.glob("data/*wind10.nc")
    ds = xr.open_dataset(file_list[0])
    for file in file_list:
        ds = xr.concat([ds, xr.open_dataset(file)],dim='time')



if __name__ == '__main__':
    print("starting..", datetime.now())

    big_tic = time.perf_counter()

    # set search params
    date_list = make_dates("2012-11-25", number=2, max=30)

    # multiprocessing to speed up. can add more threads and decrease amount of days at once if need more speed
    for j in range(len(date_list)):
        with ThreadPoolExecutor(2) as exe:
            exe.submit(get_wind_10,j,date_list[j])
            exe.submit(get_wind_100,j,date_list[j])

    big_toc = time.perf_counter()
    print(f"finished the whole shebang in {(big_toc - big_tic)/60:0.4f} minutes.")
    print(datetime.now())