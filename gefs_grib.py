# deal with gefs reanalysis grib

import numpy as np
import xarray as xr
import os
import glob
import sys


def main():
    year, month, day = sys.argv[1:4]
    # year = '2012'
    # month = '12'
    # day = '02'
    filepath = '/users/jpsotka/repos/week2-wind/data/ftp.emc.ncep.noaa.gov/GEFSv12/reanalysis/FV3_reanalysis/'+year+'/'+month+'/'+day+'/'+'gec*.f000'
    files = glob.glob(filepath)
    lats = np.arange(44,66.5,0.5)[::-1]
    lons = np.arange(220,260.5,0.5)

    ds = xr.load_dataset(files[0],engine='cfgrib',filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 'gh'})
    ds = ds.sel(latitude=lats,longitude=lons,isobaricInhPa=[700,500]).expand_dims("time")
    os.remove(files[0])

    for f in files[1:]:

        ds1 = xr.load_dataset(f,engine='cfgrib',filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 'gh'})
        ds1 = ds1.sel(latitude=lats,longitude=lons,isobaricInhPa=[700,500]).expand_dims("time")
        ds = xr.concat([ds,ds1],dim="time")

        os.remove(f)


    ds = ds.gh.sortby('time')
    
    start = str(ds.time.values[0])[:13]
    end = str(ds.time.values[-1])[:13]

    filename = 'gh-reanalysis-'+start+'-'+end+'-.nc'
    ds.to_netcdf(filename)

    return None



if __name__ == "__main__":
   main()

