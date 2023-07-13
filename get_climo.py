import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xarray as xr


def avg_obs(file, hours):
    # avg the observation data based on hours
    print('Processing obs data...')
    raw = pd.read_csv(file,sep=' ',header=0,skipinitialspace=True)
    dates = np.array(raw.Dates)
    times = np.array(raw.Times)
    datetimes = np.empty_like(dates, dtype=datetime)
    for i in range(len(dates)):
        time_str = str(times[i])
        while len(time_str) < 6:
            time_str = str(0) + time_str
        datetimes[i] = datetime.strptime(str(dates[i]) + time_str,'%Y%m%d%H%M%S')

    # resampling. note: obs are every 5 mins, resampling is forward -such that 00:00 will be the avg of 00:00 - 12:00.
    res = str(hours) + 'H'
    ds = raw.set_index(datetimes).resample(res).mean()

    ds = ds.to_xarray()
    ds = ds.drop_vars(['Dates','Times','Speed','(km/hr)','Unnamed: 5'])
    
    print('Writing...')
    ds.Wind.to_netcdf('data/bm_12.nc')


if __name__ == '__main__':
    obs = avg_obs('data/bm_obs.txt',12)
    print('Done!')