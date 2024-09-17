import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import xarray as xr


def qc_obs(file):
    # quality checks, converts to m/s
    # for wind farm observations from database
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
        if not 0 <= winds[i] < 150:  # filter out unphysical values
            raw.Wind[i] = np.nan
            counter += 1
    print(counter, 'obs removed due to unphysical value')

    # clean up and add nan for empty obs
    ds = raw.drop(['Dates','Times','Speed','(km/hr)','Unnamed: 5'],axis=1).set_index(datetimes).resample('5T').mean()
    #ds = ds.loc['2019-12-01':'2021-01-01']
    total = len(ds.Wind)
    missing = np.isnan(ds.Wind).sum()
    print(missing-counter, 'obs missing out of', total)

    # step test: difference between successive observations
    # kind of an ugly way but need to flag both values
    step_back = ds.diff().abs()
    step_forward = ds.diff(periods=-1).abs()
    rep = lambda x: np.nan if x>20 else 1  # function that will map 1 to good indices and nan to bad
    step_back = step_back.applymap(rep)
    step_forward = step_forward.applymap(rep)
    step_test = step_back * step_forward

    # now only delete flagged steps if they are associated with a nan period
    for i in np.argwhere(np.isnan(step_test.Wind.values)):
        i=i[0]
        if np.isnan(ds.Wind.values[i-1:i+2]).any():
            step_test.Wind.values[i-1:i+2] = np.nan
        elif np.isnan(ds.Wind.values[i-2:i+3]).any():
            step_test.Wind.values[i-2:i+3] = np.nan
        elif np.isnan(ds.Wind.values[i-3:i+4]).any():
            step_test.Wind.values[i-3:i+4] = np.nan
        else:
            step_test.Wind.values[i] = 1

    counter = total - step_test.sum().Wind
    ds = ds * step_test
    print(counter, 'obs removed from step test')

    # do rolling windows, calculating std. nan if <9 obs in an hour.
    rolling = ds.rolling(4,min_periods=4,center=True)
    std = rolling.std(ddof=1)  # sample standard deviation

    # filter out some std threshold
    rep = lambda x: np.nan if x==0 else 1
    std_test = std.applymap(rep)

    # for i in np.argwhere(np.isnan(std_test.Wind.values)):
    #     i=i[0]
    #     print(ds.Wind.values[i-9:i+9])
    counter = total - std_test.sum().Wind
    print(counter, 'obs removed from std test')

    ds = ds * std_test / 3.6

    return ds


def qc_raw_obs(file):
    # quality checks, converts to m/s
    # for super raw obs from bch that greg gave me (from like 2009)
    print('Processing obs data...')
    raw = pd.read_csv(file,sep=',',header=None,skipinitialspace=True)
    datetimes = np.array(raw[3])
    winds = np.array(raw[4])
    counter = 0

    for i in range(len(datetimes)):
        try:
            winds[i] = float(winds[i])
            datetimes[i] = datetime.strptime(datetimes[i],'%Y-%m-%d %H:%M:%S')
        except ValueError:
            winds[i] = np.nan
            counter += 1
            continue
        if not 0 <= winds[i] < 55:  # filter out unphysical values
            winds[i] = np.nan
            counter += 1
    print(counter, 'obs removed due to unphysical value')

    # clean up and add nan for empty obs
    winds = winds.astype(np.float64)
    datetimes = datetimes.astype(np.datetime64)
    ds = pd.DataFrame({'winds':winds},datetimes).resample('1T').mean()  # resample to make sure there arne't missing dates
    total = len(ds.winds)
    missing = np.isnan(ds.winds).sum()
    print(missing-counter, 'obs missing out of', total)

    # step test: difference between successive observations
    # kind of an ugly way but need to flag both values
    step_back = ds.diff().abs()
    step_forward = ds.diff(periods=-1).abs()
    rep = lambda x: np.nan if x>15 else 1  # function that will map 1 to good indices and nan to bad
    step_back = step_back.applymap(rep)
    step_forward = step_forward.applymap(rep)
    step_test = step_back * step_forward

    counter = total - step_test.sum().winds
    ds = ds * step_test
    print(counter, 'obs removed from step test')

    # do rolling windows, calculating std.
    rolling = ds.rolling(30,min_periods=20,center=True)
    std = rolling.std()  # sample standard deviation

    # filter out some std threshold
    rep = lambda x: np.nan if (x==0 or x>5) else 1
    std_test = std.applymap(rep)

    counter = total - std_test.sum().winds
    print(counter, 'obs removed from std test')

    ds = ds * std_test
    rep = lambda x:np.random.triangular(0.5,2.9,3.3) if (x==0) else x
    ds = ds.applymap(rep)
    ds = ds.resample('5T').mean()
    ds = ds.rename(columns={'winds':'Wind'})

    return ds


def resample_data(ds1, res=12):
    # resample 5-minute obs data
    # res is number of hours per resampling period
    origin = int((res / 2) % 24)  # modulus makes it work when res is >24
    ds1 = ds1.to_dataframe()

    res_str = str(res) + 'H'
    ds = ds1.groupby(pd.Grouper(freq=res_str,origin=datetime(2011,12,1,origin))).agg(['mean','count']).swaplevel(0,1,axis=1)
    ds = ds['mean'].where(ds['count']>=(12*res/2))  # only keep periods that have >1/2 of obs
    # for example, 12:00 point for 6 hourly will have avg from 12:00 to 17:55
    
    #ds.index += pd.DateOffset(hours=int(res / 2))
    # shift the time axis so that a 12:00 point for 6 hourly will have avg from 9:00-14:55
    # this way the time axis matches the gefs times

    print(np.isnan(ds.Wind.values).sum(), 'missing', res_str, 'periods out of', len(ds.Wind.values))
    return ds


def make_histograms(ds):
    # histograms to look at obs for each year

    groups = ds.groupby("index.year")
    bins = np.arange(20)
    fig, axes = plt.subplots(2,8,figsize=(20,7),gridspec_kw = {'wspace':0.5, 'hspace':0.35})
    
    j = 0

    for i in groups:
        data = i[1]
        ax = axes.flatten()[j]
        ax.hist(data.Wind.values,bins=bins)
        ax.set_title(str(i[0])+'\n'+str(round(np.nanmean(data.Wind.values),1)))
        ax.set_xticks(bins[::2])
        j+=1
    fig.suptitle('5-min wind speed obs by year')
    
    plt.show()
    fig.savefig('plots/histograms-spring.png')

    print('done')



if __name__ == '__main__':
    #obs = qc_obs('~/Nextcloud/thesis/bm_2024.txt')
    #obs = obs.to_xarray()
    #obs.Wind.to_netcdf('data/bm_2023_cleaned.nc')

    obs = xr.open_dataset('~/Nextcloud/thesis/bm_cleaned_all.nc')

    #obs = xr.concat([obs_og,obs],dim='index').drop_duplicates(dim='index')
    obs_summer = obs.sel(index=obs.index.dt.season=='MAM')
    make_histograms(obs_summer)
    #obs1 = xr.open_dataset('data/bm_cleaned_2009-2013.nc')
    print('Done!')