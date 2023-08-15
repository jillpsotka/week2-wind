import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import xarray as xr
from cycler import cycler
import palettable

#mpl.rcParams['axes.prop_cycle'] = cycler(color=palettable.colorbrewer.qualitative.Set2_8.mpl_colors)
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#0F2080','#F5793A','#A95AA1','#85C0F9'])
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['figure.dpi'] = 200
#mpl.rcParams['figure.figsize'] = (5,5)




def mae_plot(d):
    x_axis = np.arange(6,15,1)
    mae = []
    clim = []
    for v in d.keys():
        if v[0] == 'd':
            mae.append(abs(d[v] - d.Wind).mean())
        elif v[0] == 'c':
            clim.append(abs(d[v] - d.Wind).mean())
    plt.plot(x_axis, mae, label='GEFS raw')

    plt.plot(x_axis,clim,label='Climatology')

    plt.ylabel('MAE Daily avg')
    plt.xlabel('Forecast Day')
    plt.legend()
    plt.show()


def plot_climo():
    c = xr.open_dataset('data/climo.nc')
    c_arr = np.array([])
    for m in c.level_0:
        vals = c.Wind[m-1,:].values
        c_arr = np.concatenate([c_arr, vals[~np.isnan(vals)]])

    x_axis = np.arange(1,367)
    plt.plot(x_axis,c_arr)
    plt.ylabel('Wind km/h')
    plt.xlabel('DOY')
    plt.title('BMW Climatology 2013-2019')
    plt.show()



if __name__ == '__main__':
    plot_climo()

    ds = xr.open_dataset('data/gefs-2019-d.nc')
    mae_plot(ds)

    
