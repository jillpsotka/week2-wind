# deal with gefs reanalysis grib

import numpy as np
import xarray as xr
import os
import glob
import sys
from scipy import interpolate


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

        for f in files:
            os.remove(f)

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


def gefs_wind_interpolate(member):
    member = str(member)
    filepath = '/users/jpsotka/repos/week2-wind/data/gefs-wind-*'+member+'.nc'
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
    ds = ds.drop_dims(['longitude','latitude']).drop_vars(['number'])

    filename = files[0][:-3] + '-interpolated.nc'
    ds.to_netcdf(filename)

    return None



if __name__ == "__main__":
    gefs_z_merge(0)
    #gefs_wind_interpolate(15)