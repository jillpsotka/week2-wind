# deal with gefs reanalysis grib

import numpy as np
import xarray as xr
import os
import glob
import sys


def main():
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


def all():
    filepath = '/users/jpsotka/repos/week2-wind/data/*wind100-0.nc'
    files = glob.glob(filepath)

    ds = xr.load_dataset(files[0])

    for f in files[1:]:

        ds1 = xr.load_dataset(f)
        ds = xr.concat([ds,ds1],dim="time")


    ds = ds.sortby('time')
    start = str(ds.time.values[0])[:4]
    end = str(ds.time.values[-1])[:4]

    filename = 'wind100-'+start+'-'+end+'-0.nc'  # remember to change!!
    ds.to_netcdf(filename)

    for f in files:
        os.remove(f)

    return None



if __name__ == "__main__":
   all()