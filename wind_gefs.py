import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from datetime import datetime


def look_at_obs():
    obs = xr.open_dataset('data/bm_12.nc')


if __name__ == '__main__':
    print('starting', datetime.now())
    look_at_obs()
    
    print('done',datetime.now())