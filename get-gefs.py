# downloading gefs forecasts from archive

import s3fs
import os
from pathlib import Path

import numpy as np
import pandas as pd
import glob
import xarray as xr
import cfgrib
import sys

__author__ = "Christopher Rodell, Jill Psotka"
__email__ = "crodell@eoas.ubc.ca, jpsotka@eoas.ubc.ca"


bucket = f"noaa-gefs-pds"
XX = "00"  # initialization time
forecast_hours_1 = np.arange(141,240,3)  # for gefs, first 10 days are at 3-hr
forecast_hours_2 = np.arange(240,365,6)  # after 10 days, 6-hr
forecast_hours = np.concatenate([forecast_hours_1,forecast_hours_2])
date_range = pd.date_range("2023-05-08", "2023-12-15", freq="D")
var = "wind"  # choose wind or z
members = [int(sys.argv[1])]  # system argument for running on command line
pathname = "/Users/jpsotka/repos/week2-wind/data/"  # change this


if var == "wind":
    batch = "pgrb2bp5"
    leveltype = "heightAboveGround"
    level = 80  # 80-m winds
    short = ["u","v"]
    lats = np.arange(55.5,56.5,0.5)[::-1]
    lons = np.arange(239.5,240.5,0.5)
elif var == "z":
    batch = "pgrb2ap5"
    leveltype = "isobaricInhPa"
    level = [500,700,850,1000]  # can change levels if needed
    short = "gh"
    lats = np.arange(44,66.5,0.5)[::-1]
    lons = np.arange(220,260.5,0.5)
else:
    print('unrecognized variable. Choose wind or z')


s3 = s3fs.S3FileSystem(anon=True)
fail_count = 0
for mem in members:
    print("Downloading..")
    member = str(mem)
    member_int=int(mem * 370)

    for doi in date_range:
        # first download stuff; each forecast lead time for each day has its own file
        files = s3.ls(
            f"{bucket}/gefs.{doi.strftime('%Y%m%d')}/{XX}/atmos/{batch}/"
        )
        file_dir, file_name = files[member_int].rsplit('/', 1)  # taking only the control member gec00. Others have names gep01
        last_period_index = file_name.rfind('.')
        file_name = file_name[:last_period_index]
        for hour in forecast_hours:
            if os.path.exists(
                pathname+f"gefs/{doi.strftime('%Y%m%d')}{XX}/{file_name}.f{str(hour).zfill(3)}-{member}.grib2"):
                print(f"File exists /{doi.strftime('%Y%m%d')}{XX}/{file_name}.f{str(hour).zfill(3)}-{member}.grib2")
            else:
                try:
                    s3.download(
                        file_dir+f"/{file_name}.f{str(hour).zfill(3)}",
                    pathname+f"gefs/{doi.strftime('%Y%m%d')}{XX}/{file_name}.f{str(hour).zfill(3)}-{member}.grib2",
                    )
                except:
                    print(f"WARNING: failed download for {doi.strftime('%Y%m%d')} hour {str(hour)}")
                    fail_count += 1

        # every day, amass the files for all lead times, filter for region and variable to make smaller.
        filepath = pathname+f"gefs/{doi.strftime('%Y%m%d')}{XX}/*-{member}.grib2"
        files = glob.glob(filepath)
        if not files:
            print(f"No files downloaded for date {doi.strftime('%Y%m%d')}. Proceeding to next date")
            continue

        if leveltype == "heightAboveGround":
            ds_u = xr.open_dataset(files[0],engine='cfgrib',filter_by_keys={'typeOfLevel': leveltype, 'shortName': 'u'})
            ds_u = ds_u.sel(latitude=lats,longitude=lons).expand_dims(["time","step"])
            ds_v = xr.open_dataset(files[0],engine='cfgrib',filter_by_keys={'typeOfLevel': leveltype, 'shortName': 'v'})
            ds_v = ds_v.sel(latitude=lats,longitude=lons).expand_dims(["time","step"])
        else:
            ds = xr.open_dataset(files[0],engine='cfgrib',filter_by_keys={'typeOfLevel': leveltype, 'shortName': 'gh'})
            ds = ds.sel(latitude=lats,longitude=lons,isobaricInhPa=level).expand_dims(["time","step"])
        os.remove(files[0])

        for f in files[1:]:
            if leveltype == "heightAboveGround":
                ds = xr.open_dataset(f,engine='cfgrib',filter_by_keys={'typeOfLevel': leveltype, 'shortName': 'u'})
                ds = ds.sel(latitude=lats,longitude=lons).expand_dims(["time","step"])
                ds_u = xr.concat([ds_u,ds],dim="step")
                ds2 = xr.open_dataset(f,engine='cfgrib',filter_by_keys={'typeOfLevel': leveltype, 'shortName': 'v'})
                ds2 = ds2.sel(latitude=lats,longitude=lons).expand_dims(["time","step"])
                ds_v = xr.concat([ds_v,ds2],dim="step")

            else:
                ds1 = xr.open_dataset(f,engine='cfgrib',filter_by_keys={'typeOfLevel': leveltype, 'shortName': 'gh'})
                ds1 = ds1.sel(latitude=lats,longitude=lons,isobaricInhPa=level).expand_dims(["time","step"])
                ds = xr.concat([ds,ds1],dim="step")
            
            os.remove(f)
        
        if leveltype == "heightAboveGround":
            ds = xr.merge([ds_u,ds_v])
        ds = ds.sortby('step')

        date = str(ds.time.values[0])[:13]

        filename = pathname+'gefs-'+var+'-'+date+'-'+member+'.nc'
        ds.to_netcdf(filename)
    print(f"Done. {str(fail_count)} files of member {member} failed to download.")

