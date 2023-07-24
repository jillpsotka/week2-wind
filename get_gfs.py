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


def make_dates(start_date="2012-11-25" ,number=1, max=31):
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
    print(f"finished FastHerbie in {toc - tic:0.2f} seconds. Now putting into xarray...")

    # xarray: takes ~15 minutes per month, look into cutting more?
    ds = H.xarray(":500 mb:")
    toc2 = time.perf_counter()
    print(f"finished xarray in {(toc2 - toc)/60:0.2f} minutes. Now clipping and saving...")

    steps = np.arange(168,360,12,dtype='timedelta64[ns]')*(1000000000*60*60)  # day 7-14 every 12 hours
    lats = np.arange(45,65.5,0.5)[::-1]
    lons = np.arange(230,250.5,0.5)
    ds = ds.sel(latitude=lats,longitude=lons,step=steps)
    ds.gh.to_netcdf('data/test.nc')
    toc3 = time.perf_counter()


def get_wind_10(n, period):
    # Herbie : quick
    tic = time.perf_counter()
    H = FastHerbie(period, model="gefs_reforecast",fxx=[100,270], member=1, variable_level="ugrd_hgt")
    H2 = FastHerbie(period, model="gefs_reforecast",fxx=[100,270], member=1, variable_level="vgrd_hgt")

    toc = time.perf_counter()
    print(f"finished FastHerbie in {toc - tic:0.2f} seconds. Now putting into xarray...")

    # xarray: takes a long time :(
    steps = np.arange(144,360,12,dtype='timedelta64[ns]')*(1000000000*60*60)  # day 6-14 every 12 hours
    lats = np.arange(55.5,56.5,0.5)[::-1]
    lons = np.arange(239.5,240.5,0.5)

    u_10 = H.xarray(":UGRD:10 m")
    u_10 = u_10.sel(latitude=lats,longitude=lons,step=steps)
    u_10 = u_10.drop('heightAboveGround')  # this coordinate gets in the way
    
    print('Halfway there..', datetime.now())
    v_10 = H2.xarray(":VGRD:10 m")
    v_10 = v_10.drop('heightAboveGround')  # this coordinate gets in the way
    v_10 = v_10.sel(latitude=lats,longitude=lons,step=steps)

    toc2 = time.perf_counter()
    print(f"finished xarray in {(toc2 - toc)/60:0.2f} minutes. Now saving...")

    w10 = u_10.assign(wind10=np.sqrt(u_10.u10**2 + v_10.v10**2))
    if n <10:
        title = 'data/0' + str(n) + 'wind10-1.nc'
    else:
        title = 'data/' + str(n) + 'wind10-1.nc'
    w10.wind10.to_netcdf(title)



def get_wind_100(n, period):
    # Herbie : quick
    tic = time.perf_counter()
    H = FastHerbie(period, model="gefs_reforecast",fxx=[100,270], member=1, variable_level="ugrd_hgt")
    H2 = FastHerbie(period, model="gefs_reforecast",fxx=[100,270], member=1, variable_level="vgrd_hgt")

    toc = time.perf_counter()
    print(f"finished FastHerbie in {toc - tic:0.2f} seconds. Now putting into xarray...")

    # xarray: takes a long time :(
    steps = np.arange(144,360,12,dtype='timedelta64[ns]')*(1000000000*60*60)  # day 6-14 every 12 hours
    lats = np.arange(55.5,56.5,0.5)[::-1]
    lons = np.arange(239.5,240.5,0.5)

    u_100 = H.xarray(":UGRD:100 m")
    u_100 = u_100.sel(latitude=lats,longitude=lons,step=steps)
    u_100 = u_100.drop('heightAboveGround')
    
    print('Halfway there..', datetime.now())

    v_100 = H2.xarray(":VGRD:100 m")
    v_100 = v_100.drop('heightAboveGround')
    v_100 = v_100.sel(latitude=lats,longitude=lons,step=steps)
    toc2 = time.perf_counter()
    print(f"finished xarray in {(toc2 - toc)/60:0.2f} minutes.", datetime.now())

    w100 = u_100.assign(wind100=np.sqrt(u_100.u100**2 + v_100.v100**2))
    if n <10:
        title = 'data/0' + str(n) + 'wind100-1.nc'
    else:
        title = 'data/' + str(n) + 'wind100-1.nc'
    w100.wind100.to_netcdf(title)



def merge_data():
    #take multiple ncs of gefs and combine them
    print('merging netcdf files...')
    file_list = glob.glob("data/*wind10-1.nc")
    file_list.sort()
    ds = xr.open_dataset(file_list[0])
    for file in file_list[1:]:
        ds = xr.concat([ds, xr.open_dataset(file)],dim='time')
    
    start = str(ds.time.values[0])[:10]
    end = str(ds.time.values[-1])[:10]

    file_list = glob.glob("data/*wind100-1.nc")
    file_list.sort()
    ds2 = xr.open_dataset(file_list[0])
    for file in file_list[1:]:
        ds2 = xr.concat([ds2, xr.open_dataset(file)],dim='time')
    
    ds = xr.merge([ds,ds2])

    ds.to_netcdf('data/wind-'+start+'-'+end+'-1.nc')



if __name__ == '__main__':
    print("starting..", datetime.now())

    big_tic = time.perf_counter()

    # set search params
    date_list = make_dates("2018-12-26", number=1, max=16)

    # multiprocessing to speed up. can add more threads and decrease amount of days at once if need more speed
    for j in np.arange(stop=len(date_list)):
        with ThreadPoolExecutor(3) as exe:
            exe.submit(get_wind_10,j,date_list[j])
            exe.submit(get_wind_100,j,date_list[j])


    merge_data()

    big_toc = time.perf_counter()
    print(f"finished the whole shebang in {(big_toc - big_tic)/60:0.2f} minutes or {(big_toc - big_tic)/3600:0.2f} hours.")
    print(datetime.now())