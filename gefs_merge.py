# deal with gefs reanalysis grib

import numpy as np
import xarray as xr
import os
import glob
import sys
from scipy import interpolate
from matplotlib import pyplot as plt


def main():
    # keep this for the bash script for reforecast
    year, month = sys.argv[1:3]
    filepath = '/users/jpsotka/repos/week2-wind/data/gh-reanalysis-'+year+'-'+month+'-*.nc'
    files = glob.glob(filepath)

    ds = xr.load_dataset(files[0])
    os.remove(files[0])

    for f in files[1:]:

        ds1 = xr.load_dataset(f)
        ds = xr.concat([ds,ds1],dim="time")

        os.remove(f)


    ds = ds.gh.sortby('time')

    filename = 'gh-reanalysis-'+year+'-'+month+'.nc'
    ds.to_netcdf(filename)

    return None


def z_merge_levels(member):
    ds = xr.open_dataset('data/z-gefs/gefs-z-2020-09-26-2024-05-01-'+str(member)+'.nc')
    ds1 = xr.open_dataset('data/z-gefs/gefs-z-2020-09-26-2023-12-31-'+str(member)+'.nc')

    ds = xr.merge([ds, ds1])
    ds.to_netcdf('data/z-gefs/gefs-z-2020-09-26-2024-05-01-'+str(member)+'-n.nc')

    return None


def gefs_z_merge(member):
    member = str(member)
    filepath = '/users/jpsotka/repos/week2-wind/data/gefs-z-*-'+member+'.nc'
    files = glob.glob(filepath)

    ds = xr.load_dataset(files[0])

    for f in files[1:]:

        ds1 = xr.load_dataset(f)
        ds = xr.concat([ds,ds1],dim="time")


    ds = ds.sortby('time')
    ds = ds.drop_duplicates('time')
    start = str(ds.time.values[0])[:10]
    end = str(ds.time.values[-1])[:10]

    filename = 'data/gefs-z-'+start+'-'+end+'-'+member+'.nc'
    ds.to_netcdf(filename)

    if glob.glob(filename):
        print('successful save. deleting files..')

        # for f in files:
        #     os.remove(f)

    return None


def gefs_wind_merge(member):
    member = str(member)
    filepath = '/users/jpsotka/repos/week2-wind/data/gefs-wind-*-'+member+'.nc'
    files = glob.glob(filepath)

    ds = xr.load_dataset(files[0])

    for f in files[1:]:

        ds1 = xr.load_dataset(f)
        ds = xr.concat([ds,ds1],dim="time")


    ds = ds.sortby('time')
    ds = ds.drop_duplicates('time')
    start = str(ds.time.values[0])[:10]
    end = str(ds.time.values[-1])[:10]

    filename = 'data/gefs-wind-'+start+'-'+end+'-'+member+'.nc'
    ds.to_netcdf(filename)

    if glob.glob(filename):
        print('successful save. deleting files..')

        for f in files:
            os.remove(f)

    return None


def gefs_z_merge_members():
    ds = xr.open_dataset('data/gefs-z-2020-09-25-2023-12-31-1-3-merged.nc')
    #ds = ds.assign_coords(member=1)
    ds1 = xr.open_dataset('data/gefs-z-2020-09-26-2023-12-31-4.nc')
    ds1 = ds1.assign_coords(member=4)

    ds = xr.concat([ds,ds1],dim='member')

    ds.to_netcdf('data/gefs-z-2020-09-25-2023-12-31-1-3-4-merged.nc')


def gefs_wind_merge_members():
    filepath = '/users/jpsotka/repos/week2-wind/data/gefs-wind-*-interpolated.nc'
    files = glob.glob(filepath)
    files.sort()
    if len(files) <= 1:
        print(len(files)+' file(s) found - mistake?')
        sys.exit()
    ds = xr.open_dataset(files[0])
    mem = files[0].split('-')[-2]
    try:  # got one of the member files
        mem = int(mem)
        ds = ds.assign_coords(member=mem)
    except ValueError:  # got the master file
        mem = None

    try:
        ds = ds.drop_vars(['valid_time'])
    except:
        None
    
    if 'u' in list(ds.keys()):  # calculate wind speed instead of u and v
        ds['wind'] = (['time','step'],np.sqrt(ds['u'].values**2 + ds['v'].values**2))
        ds = ds.drop_vars(['u','v'])
    
    for f in files[1:]:
        ds1 = xr.load_dataset(f)
        if 'u' in list(ds1.keys()):  # calculate wind speed instead of u and v
            ds1['wind'] = (['time','step'],np.sqrt(ds1['u'].values**2 + ds1['v'].values**2))
            ds1 = ds1.drop_vars(['u','v'])
        mem = f.split('-')[-2]
        try:  # got one of the member files
            mem = int(mem)
            ds1 = ds1.assign_coords(member=mem)
        except:  # got the master file
            mem = None
        try:
            ds1 = ds1.drop_vars(['valid_time'])
        except:
            None
        if type(mem) == int:
            ds = xr.concat([ds,ds1],dim="member")
        else:
            ds = xr.concat([ds1,ds],dim="time")

    ds.to_netcdf('data/gefs-wind-all-interpolated-a.nc')



def gefs_wind_interpolate(member):
    member = str(member)
    filepath = '/users/jpsotka/repos/week2-wind/data/gefs-wind-*-'+member+'.nc'
    files = glob.glob(filepath)

    if len(files) > 1:
        print('Multiple files found. Merge files first.')
        sys.exit()

    ds = xr.load_dataset(files[0])

    # horizontal interpolation
    coords = np.array([[ds.latitude.values[0],ds.longitude.values[0]],  # (56,239.5)
                       [ds.latitude.values[0],ds.longitude.values[1]],
                       [ds.latitude.values[1],ds.longitude.values[0]],
                       [ds.latitude.values[1],ds.longitude.values[1]]])
    loc = np.array([55.6986, 360-120.4306])  # insert lat/lon of wind farm
    u = np.array([ds.u.isel(latitude=0,longitude=0),
                       ds.u.isel(latitude=0,longitude=1),
                       ds.u.isel(latitude=1,longitude=0),
                       ds.u.isel(latitude=1,longitude=1)])
    
    u = interpolate.griddata(coords, u, loc, method='linear').squeeze() 

    v = np.array([ds.v.isel(latitude=0,longitude=0),
                       ds.v.isel(latitude=0,longitude=1),
                       ds.v.isel(latitude=1,longitude=0),
                       ds.v.isel(latitude=1,longitude=1)])
    
    v = interpolate.griddata(coords, v, loc, method='linear').squeeze() 

    ds = ds.assign(u=(['time','step'],u))
    ds = ds.assign(v=(['time','step'],v))
    ds = ds.drop_dims(['longitude','latitude']).drop_vars(['number','valid_time'])

    filename = files[0][:-3] + '-interpolated.nc'
    ds.to_netcdf(filename)

    return None



if __name__ == "__main__":
    #z_merge_levels(12)
    gefs_z_merge(17)
    #gefs_z_merge(18)
    #gefs_z_merge_members()
    #gefs_wind_interpolate(9)
    #gefs_wind_interpolate(10)
    #gefs_wind_merge_members()
