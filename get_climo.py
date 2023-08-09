import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
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


def qc_obs(file):
    print('Processing obs data...')
    raw = pd.read_csv(file,sep=' ',header=0,skipinitialspace=True)
    dates = np.array(raw.Dates)
    times = np.array(raw.Times)
    winds = np.array(raw.Wind)
    counter = 0

    # make pandas index datetimes
    datetimes = np.empty_like(dates, dtype=datetime)
    for i in range(len(dates)):
        time_str = str(times[i])
        while len(time_str) < 6:
            time_str = str(0) + time_str
        datetimes[i] = datetime.strptime(str(dates[i]) + time_str,'%Y%m%d%H%M%S')
        if not 0 < winds[i] < 160:  # filter out unphysical values
            raw.Wind[i] = np.nan
            counter += 1
    print(counter, 'obs removed due to unphysical value')

    # clean up and add nan for empty obs
    ds = raw.drop(['Dates','Times','Speed','(km/hr)','Unnamed: 5'],axis=1).set_index(datetimes).resample('5T').mean()
    total = len(ds.Wind)
    missing = np.isnan(ds.Wind).sum()
    print(missing-counter, 'obs missing out of', total)

    # step test: difference between successive observations
    # kind of an ugly way but need to flag both values
    step_back = ds.diff().abs()
    step_forward = ds.diff(periods=-1).abs()
    rep = lambda x: np.nan if x>30 else 1  # function that will map 1 to good indices and nan to bad
    step_back = step_back.applymap(rep)
    step_forward = step_forward.applymap(rep)
    step_test = step_back * step_forward

    counter = total - step_test.sum().Wind
    ds = ds * step_test
    print(counter, 'obs removed from step test')

    # do rolling windows, calculating std and mean per hour. nan if <9 obs in an hour.
    rolling = ds.rolling(6,min_periods=4,center=True)
    std = rolling.std()

    # filter out some std threshold
    rep = lambda x: np.nan if x<0.1 else 1
    std_test = std.applymap(rep)
    counter = total - std_test.sum().Wind
    print(counter, 'obs removed from std test')

    ds = ds * std_test

    # 12h sampling, getting rid of ones that have <1 hour of obs
    res = '12H'
    #ds = ds.resample(res).mean(skina=False)
    ds = ds.groupby(pd.Grouper(freq=res)).agg(['mean','count']).swaplevel(0,1,axis=1)
    ds = ds['mean'].where(ds['count']>=12)

    ds = ds.to_xarray()

    print('done')


def make_histograms(file):
    # histograms to look at obs for each year
    raw = pd.read_csv(file,sep=' ',header=0,skipinitialspace=True)
    dates = np.array(raw.Dates)
    times = np.array(raw.Times)
    winds = np.array(raw.Wind)

    # make pandas index datetimes
    datetimes = np.empty_like(dates, dtype=datetime)
    for i in range(len(dates)):
        time_str = str(times[i])
        while len(time_str) < 6:
            time_str = str(0) + time_str
        datetimes[i] = datetime.strptime(str(dates[i]) + time_str,'%Y%m%d%H%M%S')
        if not 0 < winds[i] < 160:  # filter out unphysical values
            raw.Wind[i] = np.nan

    ds = raw.drop(['Dates','Times','Speed','(km/hr)','Unnamed: 5'],axis=1).set_index(datetimes)
    years = ds.groupby(pd.Grouper(freq="Y"))
    ax = years.hist(bins=np.arange(0,100,10))
    title = 2012
    for x in ax:
        x[0][0].set_xlabel('wind speed (km/h)')
        x[0][0].set_ylabel('counts')
        x[0][0].set_title(str(title))
        title += 1

    print('done')


if __name__ == '__main__':
    obs = qc_obs('data/test_obs.txt')
    #make_histograms('data/bm_obs.txt')
    print('Done!')