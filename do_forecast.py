import numpy as np
import pickle
import pandas as pd
from fitter import Fitter, get_common_distributions, get_distributions
from som_class import SOM, BMUs, BMU_frequency
from matplotlib import pyplot as plt
import xarray as xr
from resampling import gefs_reanalysis, cleaned_obs, dates_obs_gefs
from do_som import prep_data
import scipy.stats
import xskillscore as xs



def wind_distributions(bmus):
    
    distributions = []
    
    fig, axes = plt.subplots(nrows=Ny, ncols=Nx, gridspec_kw = {'wspace':0.5, 'hspace':0.5})
    vmin = np.min(obs)
    vmax = np.max(obs)
    
    for i, ax in enumerate(axes.flatten()):
        distribution = obs[np.where(bmus==i)[0]]  # wind obs that belong to this node -> this is our distribution
        ax.hist(distribution, range=(vmin,vmax),bins='auto')
        ax.set_title('Avg wind speed ='+str(round(np.mean(distribution),2))+'m/s')
        distributions.append(distribution)

    #plt.tight_layout()
    plt.show()

    with open('distributions.pkl','wb') as f:
        pickle.dump(distributions,f)

    return distributions


def fit_hist(ds):
    # actually, maybe dont want to fit. can just use the distribution of obs?
    f = Fitter(ds, distributions=['beta','geninvgauss','chi2','ncx2','lognorm','nakagami'])
    f.fit()
    print(f.summary())

    dist = scipy.stats.nakagami(nu=0.765176, loc=1.6762, scale=4.758206) # from fitter
    dist.cdf(5) # this will give % of obs under 5m/s


    return None


def cdf(x):
    dist = distributions[0]
    dist1 = dist.where(dist[0]<x)
    val = len(dist1)/len(dist)

    return val


if __name__ ==  "__main__":
    # obs = cleaned_obs(res=6)
    # gefs = xr.open_dataset('gh-reanalysis-all-2012-2019.nc').sel(isobaricInhPa=500)
    # gefs = gefs_reanalysis(gefs)
    # obs, gefs = dates_obs_gefs(obs, gefs)
    # obs, gefs = prep_data(gefs, obs)

    # with open('trained-map.pkl','rb') as handle:
    #     som = pickle.load(handle)
    # bmus = BMUs(som)
    # Nx = 6
    # Ny = 3
    # N_nodes = Nx * Ny

    #wind_distributions(bmus)
    with open('distributions.pkl','rb') as handle:
        distributions = pickle.load(handle)

    fit_hist(distributions[1])
    y=[4,6.1,3.2,7,5,5.5]

    crps = xs.crps_quadrature(y,cdf)
    print(crps)


    