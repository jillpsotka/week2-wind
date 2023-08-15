from herbie import FastHerbie
import numpy as np
import pandas as pd
import xarray as xr
import time
#import cartopy.crs as ccrs
import glob
#from toolbox import EasyMap, pc
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


def make_dates(start_date="2012-11-25" ,number=23, max=16):
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


def get_wind_10(n, period, mem,lat,lon,dir=''):
    # Herbie : quick
    tic = time.perf_counter()
    H = FastHerbie(period, model="gefs_reforecast",fxx=[100,270], member=mem, variable_level="ugrd_hgt")
    H2 = FastHerbie(period, model="gefs_reforecast",fxx=[100,270], member=mem, variable_level="vgrd_hgt")

    toc = time.perf_counter()
    print(f"finished FastHerbie in {toc - tic:0.2f} seconds. Now putting into xarray...")

    # xarray: takes a long time :(
    u_10 = H.xarray(":UGRD:10 m",lats=lat,lons=lon)
    if len(u_10.time) > len(period):  # something went wrong (e.g. nov 9 2019)
        s1 = len(u_10.step)
        u_10 = u_10.sel(time=period).isel(step=slice(int(s1/2),s1))
        print('problem date in', period)

    steps = np.arange(144,240,3,dtype='timedelta64[ns]')*(1000000000*60*60)  # day 6-10 every 3 hours
    steps2 = np.arange(240,361,6,dtype='timedelta64[ns]')*(1000000000*60*60)  # day 10-15 every 6 hours
    steps = np.concatenate([steps,steps2])

    u_10 = u_10.sel(latitude=lat,longitude=lon,step=steps)
    u_10 = u_10.drop('heightAboveGround')  # this coordinate gets in the way
    
    print('Halfway there..', datetime.now())
    v_10 = H2.xarray(":VGRD:10 m",lats=lat,lons=lon)
    v_10 = v_10.drop('heightAboveGround')  # this coordinate gets in the way
    if len(v_10.time) > len(period):  # something went wrong (e.g. nov 9 2019)
        s1 = len(v_10.step)
        v_10 = v_10.sel(time=period).isel(step=slice(int(s1/2),s1))
    v_10 = v_10.sel(latitude=lat,longitude=lon,step=steps)

    toc2 = time.perf_counter()
    print(f"finished xarray in {(toc2 - toc)/60:0.2f} minutes. Now saving...")

    w10 = u_10.assign(wind10=np.sqrt(u_10.u10**2 + v_10.v10**2))
    if n <10:
        title = dir + '0' + str(n) + 'wind10-' + str(mem) + '.nc'
    else:
        title = dir + str(n) + 'wind10-' + str(mem) + '.nc'
    w10.wind10.to_netcdf(title)
    print('Done that set', datetime.now())


def get_wind_100(n, period,mem,lat,lon,dir=''):
    # Herbie : quick
    tic = time.perf_counter()
    H = FastHerbie(period, model="gefs_reforecast",fxx=[100,270], member=mem, variable_level="ugrd_hgt")
    H2 = FastHerbie(period, model="gefs_reforecast",fxx=[100,270], member=mem, variable_level="vgrd_hgt")

    toc = time.perf_counter()
    print(f"finished FastHerbie in {toc - tic:0.2f} seconds. Now putting into xarray...")

    # xarray: takes a long time :(
    u_100 = H.xarray(":UGRD:100 m",lats=lat,lons=lon)
    if len(u_100.time) > len(period):  # something went wrong (e.g. nov 9 2019)
        s1 = len(u_100.step)
        u_100 = u_100.sel(time=period).isel(step=slice(int(s1/2),s1))
    
    steps = np.arange(144,240,3,dtype='timedelta64[ns]')*(1000000000*60*60)  # day 6-10 every 3 hours
    steps2 = np.arange(240,361,6,dtype='timedelta64[ns]')*(1000000000*60*60)  # day 10-15 every 6 hours
    steps = np.concatenate([steps,steps2])

    u_100 = u_100.sel(latitude=lat,longitude=lon,step=steps)
    u_100 = u_100.drop('heightAboveGround')
    
    print('Halfway there..', datetime.now())

    v_100 = H2.xarray(":VGRD:100 m",lats=lat,lons=lon)
    v_100 = v_100.drop('heightAboveGround')
    if len(v_100.time) > len(period):  # something went wrong (e.g. nov 9 2019)
        s1 = len(v_100.step)
        v_100 = v_100.sel(time=period).isel(step=slice(int(s1/2),s1))
    v_100 = v_100.sel(latitude=lat,longitude=lon,step=steps)
    toc2 = time.perf_counter()
    print(f"finished xarray in {(toc2 - toc)/60:0.2f} minutes.", datetime.now())

    w100 = u_100.assign(wind100=np.sqrt(u_100.u100**2 + v_100.v100**2))
    if n <10:
        title = dir + '0' + str(n) + 'wind100-' + str(mem) + '.nc'
    else:
        title = dir + str(n) + 'wind100-' + str(mem) + '.nc'
    w100.wind100.to_netcdf(title)
    print('Done that set', datetime.now())


def do_all_wind(n, period,mem,lat,lon,dir=''):
    get_wind_10(n,period,mem,lat,lon,dir)
    print('Done 10m')
    get_wind_100(n,period,mem,lat,lon,dir)


def merge_data(member=0,dir=''):
    #take multiple ncs of gefs and combine them
    print('merging netcdf files...')
    file_list = glob.glob(dir + "*wind10-"+str(member)+".nc")
    file_list.sort()
    ds = xr.open_dataset(file_list[0])
    for file in file_list[1:]:
        ds = xr.concat([ds, xr.open_dataset(file)],dim='time')
    
    start = str(ds.time.values[0])[:10]
    end = str(ds.time.values[-1])[:10]

    file_list = glob.glob(dir + "*wind100-"+str(member)+".nc")
    file_list.sort()
    ds2 = xr.open_dataset(file_list[0])
    for file in file_list[1:]:
        ds2 = xr.concat([ds2, xr.open_dataset(file)],dim='time')
    
    ds = xr.merge([ds,ds2])

    ds.to_netcdf(dir + 'wind-'+start+'-'+end+'-'+str(member)+'.nc')


if __name__ == '__main__':
    print("starting..", datetime.now())

    big_tic = time.perf_counter()

    # set search params
    chunks = 53 #23
    days = 7  #16
    # nov 9 has to be on day 1 (?)
    date_list = make_dates("2018-12-22", number=chunks, max=days)  # 2018-12-26 start
    member = 4
    lats = np.arange(55.5,56.5,0.5)[::-1]
    lons = np.arange(239.5,240.5,0.5)
    dir = "data/"

    pool = multiprocessing.Pool(3)
    processes = [pool.apply_async(do_all_wind, args=(j, date_list[j],member,lats,lons,dir))for j in range(len(date_list))]
    result = [p.get() for p in processes]

    merge_data(member,dir)

    big_toc = time.perf_counter()
    print(f"finished the whole shebang in {(big_toc - big_tic)/60:0.2f} minutes or {(big_toc - big_tic)/3600:0.2f} hours.")
    print(datetime.now())