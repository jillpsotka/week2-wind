import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import xarray as xr



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
        if not 0 <= winds[i] < 150:  # filter out unphysical values
            raw.Wind[i] = np.nan
            counter += 1
    print(counter, 'obs removed due to unphysical value')

    # clean up and add nan for empty obs
    ds = raw.drop(['Dates','Times','Speed','(km/hr)','Unnamed: 5'],axis=1).set_index(datetimes).resample('5T').mean()
    ds = ds.loc['2012-12-01':'2020-01-01']
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
    rolling = ds.rolling(6,min_periods=6,center=True)
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


def resample_data(ds1, res=12):
    # res is number of hours per resampling period
    #ds = xr.open_dataset(ds_str)
    #ds1 = ds1.isel(index=slice(0,300))
    ds1 = ds1.to_dataframe()

    res_str = str(res) + 'H'
    ds = ds1.groupby(pd.Grouper(freq=res_str,origin=datetime(2011,12,1,3))).agg(['mean','count']).swaplevel(0,1,axis=1)
    ds = ds['mean'].where(ds['count']>=(12*res/2))  # only keep periods that have >1/2 of obs
    # for example, 12:00 point for 6 hourly will have avg from 12:00 to 17:55

    #times = pd.date_range(datetime(2012,12,1,12),datetime(2012,12,1,18),freq='5T')

    print(np.isnan(ds.Wind.values).sum(), 'missing', res_str, 'periods out of', len(ds.Wind.values))
    return ds


def make_climo_daily(ds):
    ds = ds.loc['2012-12-01':'2018-12-31']

    c = ds.groupby([ds.index.month, ds.index.day]).mean()  # group by DOY
    c = pd.concat([c[-15:],c,c[:15]])  # repeat beginning and end of year for rolling stuff
    c.index.rename('day',level=1,inplace=True)
    c.index.rename('month',level=0,inplace=True)

    days = c.Wind
    index_d = np.where(days.isna())
    for i in index_d[0]:
        days.iloc[i] = days.iloc[i-1:i+2].mean()  # set feb 29 to be avg of feb 27 and mar 1

    days_rolling = days.rolling(21,min_periods=21,center=True).mean()  # rolling avg 10 days

    days_rolling = days_rolling.rolling(11,min_periods=11,center=True).mean()
    
    days_rolling = days_rolling[days_rolling.notna()]

    plt.plot(days_rolling.values,label='day')
    plt.title('BMW Climatology 2013-2019')
    plt.xlabel('Day of year')
    plt.ylabel('Wind speed (m/s)')
    plt.show()

    days_rolling.rename("daytime",inplace=True)

    return days_rolling



def make_climo_12(ds):
    ds = ds.loc['2012-12-01':'2018-12-31']

    c = ds.groupby([ds.index.month, ds.index.day, ds.index.hour]).mean()  # group by DOY
    c = pd.concat([c[-30:],c,c[:30]])  # repeat beginning and end of year for rolling stuff
    c.index.rename('day',level=1,inplace=True)
    c.index.rename('month',level=0,inplace=True)
    c.index.rename('hour',level=2,inplace=True)
    c = c.reset_index(level='hour')

    days = c.loc[c['hour']>4,'Wind']
    index_d = np.where(days.isna())
    for i in index_d[0]:
        days.iloc[i] = days.iloc[i-1:i+2].mean()  # set feb 29 to be avg of feb 27 and mar 1
    nights = c.loc[c['hour']<4,'Wind']  # because of time difference, this is the night BEFORE
    index_n = np.where(nights.isna())
    for i in index_n[0]:
        nights.iloc[i] = nights.iloc[i-1:i+2].mean()  # set feb 29 to be avg of feb 27 and mar 1

    days_rolling = days.rolling(21,min_periods=21,center=True).mean()  # rolling avg 10 days
    nights_rolling = nights.rolling(21,min_periods=21,center=True).mean()

    days_rolling = days_rolling.rolling(11,min_periods=11,center=True).mean()
    nights_rolling = nights_rolling.rolling(11,min_periods=11,center=True).mean()
    
    days_rolling = days_rolling[days_rolling.notna()]
    nights_rolling = nights_rolling[nights_rolling.notna()]

    plt.plot(days_rolling.values,label='day')
    plt.plot(nights_rolling.values, label='night')
    plt.title('BMW Climatology 2013-2019')
    plt.xlabel('Day of year')
    plt.ylabel('Wind speed (m/s)')
    plt.legend()
    plt.show()

    days_rolling.rename("daytime",inplace=True)
    nights_rolling.rename("nighttime",inplace=True)
    c = pd.concat([days_rolling,nights_rolling],axis=1)

    return c


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
    # obs = qc_obs('data/bm_obs.txt')
    # obs = obs.to_xarray()
    # obs.Wind.to_netcdf('data/bm_cleaned.nc')

    #obs = resample_data('data/bm_cleaned.nc')
    obs = xr.open_dataset('data/bm_cleaned.nc')
    obs = resample_data(obs,24)

    climo = make_climo_daily(obs)
    #make_histograms('data/bm_obs.txt')

    climo = climo.to_xarray()
    #climo.to_netcdf('data/climo-daily.nc')

    print('Done!')