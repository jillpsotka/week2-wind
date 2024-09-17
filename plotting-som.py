import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import xarray as xr
import pandas as pd
from scipy import signal
from resampling import resample_mean
import pickle
import properscoring as ps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cmcrameri import cm
import matplotlib.cm as comap
import sys
import warnings
warnings.filterwarnings("ignore")
import matplotlib.patheffects as pe


# testing and plotting

mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['figure.dpi'] = 250
mpl.rcParams['image.cmap'] = 'cmc.lipari'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#0072B2","#CC79A7","#BFBFBF","#4268B3"])


def era_stats(aggression):
    if aggression == 0:
        thresh = 0
    elif aggression == 1:
        thresh = 0.1
    elif aggression == 2:
        thresh = 0.25
    crps_som = np.empty((N_nodes,len(obs_val.Wind.values)))  # (nodes, forecasts)
    crps_som.fill(np.nan)
    crps_clim = np.empty((N_nodes,len(obs_val.Wind.values)))
    crps_clim.fill(np.nan)
    for kk, gh in enumerate(erav.gh):  # for each validation date
        # find the bmu and record the crps for this date
        if np.isnan(gh.any()):
            continue
        else:
            mem_arr = np.reshape(gh.to_numpy(),(gh.latitude.shape[0]*gh.longitude.shape[0]))
            BMU = np.argmin(np.linalg.norm(mem_arr - som.z_raw, axis=1))
            ob = obs_val.sel(index=gh.time.values).Wind
            crps_som[BMU,kk] = ps.crps_ensemble(ob, distributions[BMU])
            crps_clim[BMU,kk] =ps.crps_ensemble(ob, clim_prob.sel(index=clim_prob.index.hour==gh.time.dt.hour).Wind)

    bad_nodes = []
    for j,d in enumerate(distributions):  # for each node
        # calculate total crpss for each node
        crpss = np.nanmedian(1-crps_som[j,:]/crps_clim[j,:])
        if crpss < thresh:  # flag indices of 'bad' nodes
            bad_nodes.append(j)

    return bad_nodes



def get_climo_det(ds):
    # this makes deterministic climo sorted by hour
    ds = ds.assign_coords(
        {
            #"doy": ds["index.dayofyear"],
            "hour": ds["index.hour"]
        }
    )
    clim_prob = ds  # get hour by   clim_prob.sel(index=clim_prob.index.hour==0)
    clim_det = ds.groupby('index.hour').mean()

    return clim_prob, clim_det


def ens_members(gefs):
    x_axis = pd.date_range('2019-01-01','2019-02-01')
    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0,member=0).wind,linewidth=0.4,c=colors[2])
    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0,member=1).wind,linewidth=0.4,c=colors[2])
    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0,member=2).wind,linewidth=0.4,c=colors[2])
    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0,member=3).wind,linewidth=0.4,c=colors[2])
    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0,member=4).wind,linewidth=0.4,c=colors[2])
    

    plt.plot(x_axis, gefs.sel(time=x_axis).isel(step=0).wind.mean(dim='member'),linewidth=1,c='black')

    lower = gefs.sel(time=x_axis).isel(step=0).wind.min(dim='member')
    upper = gefs.sel(time=x_axis).isel(step=0).wind.max(dim='member')
    plt.fill_between(x_axis, lower, upper, color='cyan', alpha=0.2)

    plt.xticks(rotation = 45)
    plt.ylabel('Wind speed (m/s)')
    plt.show()

def plot_som_6x4(Nx1, Ny1, z, indices,bad_nodes):
    proj=ccrs.PlateCarree()
    vmin = np.min(z)
    vmax = np.max(z)  # colorbar range
    Nx = Ny1
    Ny = Nx1
    fig, axes = plt.subplots(nrows=Ny, ncols=Nx,sharex=True,sharey='row',layout='constrained',figsize=(Nx*2,Ny*1.8),
                             subplot_kw={'projection': proj, 'aspect':1.7},gridspec_kw = {'wspace':0.01, 'hspace':0.08})
    im = comap.ScalarMappable(norm=Normalize(vmin,vmax),cmap='cmc.lipari')
    for kk, ax in enumerate(axes.flatten()):
        if lat_offset == 9:
            var = z[indices[kk],:].reshape(37,65)
            ax.set_extent(([223,257,45.5,63.65]))
        else:
            var = z[indices[kk],:].reshape(45,81)
            ax.set_extent(([219,261,43.25,65.25]))
        ax.set_title('Node '+str(indices[kk]+1),fontsize=15)      
        
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_0_boundary_lines_land', scale='50m', facecolor='none', edgecolor='k',alpha=0.6,linewidth=0.6))
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_1_states_provinces_lines', scale='50m',facecolor='none', edgecolor='k',alpha=0.6,linewidth=0.6))
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', \
            scale='50m', edgecolor='k', facecolor='none', alpha=0.6,linewidth=0.5))
        
        cs = ax.contourf(lon, lat, var, vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),cmap='cmc.lipari')
        ax.scatter(360-120.4306,55.6986,c='yellow',transform=ccrs.PlateCarree(),s=5,marker='*',zorder=20)

        # Create gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.8, color='black', alpha=0.2,linestyle='--')
        # Manipulate gridlines number and spaces
        gl.ylocator = mticker.FixedLocator(np.arange(-90,90,10))
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10))

        if indices[kk] in bad_nodes:
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='grey', linewidth=3, alpha=0.7,
                    path_effects=[pe.Stroke(linewidth=4, foreground='k',alpha=0.7), pe.Normal()],zorder=21)
            ax.plot([0, 1], [1, 0], transform=ax.transAxes, color='grey', linewidth=3, alpha=0.7,
                    path_effects=[pe.Stroke(linewidth=4, foreground='k',alpha=0.7), pe.Normal()],zorder=21)

    #cbar_ax = fig.add_axes([0.05, 0.07, 0.45*Nx, 0.03])
    cbar = fig.colorbar(im,ax=axes,fraction=0.045, shrink=0.65,pad=0.05,location='right')
    cbar.set_label('850-hPa Geopotential Height Anomaly (m)',size=16)
    cbar.ax.tick_params(labelsize=14)

    #plt.suptitle('z500 clusters')
    plt.savefig('plots/som-'+seas+'-2x2-final.pdf',bbox_inches='tight')

    return None

def plot_som(Nx1, Ny1, z, indices,bad_nodes):
    proj=ccrs.PlateCarree()
    vmin = np.min(z)
    vmax = np.max(z)  # colorbar range
    Nx = Ny1
    Ny = Nx1
    fig, axes = plt.subplots(nrows=int(Ny/2+1), ncols=int(Nx*2+1),sharex=True,sharey='row',figsize=(4,6),layout='constrained',
                             subplot_kw={'projection': proj, 'aspect':1.7},gridspec_kw = {'width_ratios':[0.5,0.5,0.2,0.5,0.5]})
    im = comap.ScalarMappable(norm=Normalize(vmin,vmax),cmap='cmc.lipari')
    for kk, ax in enumerate(axes.flatten()):
        if (kk-2) % 5 == 0:
            ax.axis('off')
            continue
        elif kk > 37:
            ax.axis('off')
            continue
        elif kk > 2:
            kk -= ((kk-3) // 5)+1
        if lat_offset == 9:
            var = z[indices[kk],:].reshape(37,65)
            ax.set_extent(([223,257,45.5,63.65]))
        else:
            var = z[indices[kk],:].reshape(45,81)
            ax.set_extent(([219,261,43.25,65.25]))
        ax.set_title('Node '+str(indices[kk]+1),fontsize=8,pad=1.3)      
        
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_0_boundary_lines_land', scale='50m', facecolor='none', edgecolor='k',alpha=0.6,linewidth=0.6))
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_1_states_provinces_lines', scale='50m',facecolor='none', edgecolor='k',alpha=0.6,linewidth=0.6))
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', \
            scale='50m', edgecolor='k', facecolor='none', alpha=0.6,linewidth=0.5))
        
        cs = ax.contourf(lon, lat, var, vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),cmap='cmc.lipari')
        ax.scatter(360-120.4306,55.6986,c='yellow',transform=ccrs.PlateCarree(),s=5,marker='*',zorder=20)

        # Create gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.8, color='black', alpha=0.2,linestyle='--')
        # Manipulate gridlines number and spaces
        gl.ylocator = mticker.FixedLocator(np.arange(-90,90,10))
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10))

        if indices[kk] in bad_nodes:
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='grey', linewidth=1, alpha=0.7,
                    path_effects=[pe.Stroke(linewidth=2, foreground='k',alpha=0.7), pe.Normal()],zorder=21)
            ax.plot([0, 1], [1, 0], transform=ax.transAxes, color='grey', linewidth=1, alpha=0.7,
                    path_effects=[pe.Stroke(linewidth=2, foreground='k',alpha=0.7), pe.Normal()],zorder=21)

    cbar = fig.colorbar(im,ax=axes,fraction=0.045, shrink=0.65,pad=0.04,location='right')
    if level == 1000:
        cbar.set_label('1000-hPa Geopotential Height Anomaly (m)',size=9)
    else:
        cbar.set_label('850-hPa Geopotential Height Anomaly (m)',size=9)
    cbar.ax.tick_params(labelsize=8)

    #plt.suptitle('z500 clusters')
    plt.savefig('plots/som-'+seas+'-2x2-final.pdf',bbox_inches='tight')

    return None

def plot_som_14x2(Nx1, Ny1, z, indices,bad_nodes):
    proj=ccrs.PlateCarree()
    vmin = np.min(z)
    vmax = np.max(z)  # colorbar range
    Nx = Ny1
    Ny = Nx1
    fig, axes = plt.subplots(nrows=int(Ny/2), ncols=int(Nx*2+1),sharex=True,sharey='row',figsize=(4,5.5),layout='constrained',
                             subplot_kw={'projection': proj, 'aspect':1.7},gridspec_kw = {'width_ratios':[0.5,0.5,0.2,0.5,0.5]})
    im = comap.ScalarMappable(norm=Normalize(vmin,vmax),cmap='cmc.lipari')
    for kk, ax in enumerate(axes.flatten()):
        if (kk-2) % 5 == 0:
            ax.axis('off')
            continue
        elif kk > 2:
            kk -= ((kk-3) // 5)+1
        if lat_offset == 9:
            var = z[indices[kk],:].reshape(37,65)
            ax.set_extent(([223,257,45.5,63.65]))
        else:
            var = z[indices[kk],:].reshape(45,81)
            ax.set_extent(([219,261,43.25,65.25]))
        ax.set_title('Node '+str(indices[kk]+1),fontsize=8,pad=1.3)      
        
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_0_boundary_lines_land', scale='50m', facecolor='none', edgecolor='k',alpha=0.6,linewidth=0.6))
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
            name='admin_1_states_provinces_lines', scale='50m',facecolor='none', edgecolor='k',alpha=0.6,linewidth=0.6))
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', \
            scale='50m', edgecolor='k', facecolor='none', alpha=0.6,linewidth=0.5))
        
        cs = ax.contourf(lon, lat, var, vmin=vmin,vmax=vmax, transform=ccrs.PlateCarree(),cmap='cmc.lipari')
        ax.scatter(360-120.4306,55.6986,c='yellow',transform=ccrs.PlateCarree(),s=5,marker='*',zorder=20)

        # Create gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.8, color='black', alpha=0.2,linestyle='--')
        # Manipulate gridlines number and spaces
        gl.ylocator = mticker.FixedLocator(np.arange(-90,90,10))
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10))
        if indices[kk] in bad_nodes:
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='grey', linewidth=1.5, alpha=0.7,
                    path_effects=[pe.Stroke(linewidth=2.5, foreground='k',alpha=0.7), pe.Normal()],zorder=21)
            ax.plot([0, 1], [1, 0], transform=ax.transAxes, color='grey', linewidth=1.5, alpha=0.7,
                    path_effects=[pe.Stroke(linewidth=2.5, foreground='k',alpha=0.7), pe.Normal()],zorder=21)

    cbar = fig.colorbar(im,ax=axes,fraction=0.045, shrink=0.65,pad=0.05,location='right')
    if level == 1000:
        cbar.set_label('1000-hPa Geopotential Height Anomaly (m)',size=9)
    else:
        cbar.set_label('850-hPa Geopotential Height Anomaly (m)',size=9)
    cbar.ax.tick_params(labelsize=9)

    #plt.suptitle('z500 clusters')
    plt.savefig('plots/som-'+seas+'-2x2-final.pdf',bbox_inches='tight')

    return None

 
def plot_distributions(distributions,bad_nodes):
    if seas == 'DJF':
        a=48 # aspect ratio
    else:
        a=35
        
    fig, axes = plt.subplots(nrows=int(Nx/2+1), ncols=int(Ny*2+1), figsize=(3.73,7),sharex=True,sharey=True,
                             subplot_kw={'aspect':a},gridspec_kw = {'width_ratios':[0.5,0.5,0.15,0.5,0.5]})
    vmin = np.nanmin(obs_train.Wind.values)
    vmax = np.nanmax(obs_train.Wind.values)
    means = []
    
    for i, ax in enumerate(axes.flatten()):
        if (i-2) % 5 == 0:
            ax.axis('off')
            continue
        elif i > 37:
            ax.axis('off')
            continue
        elif i > 2:
            i -= ((i-3) // 5)+1
        distribution = distributions[indices[i]]  # wind obs that belong to this node -> this is our distribution
        ax.hist(distribution, range=(vmin,vmax),color='black',
                weights=np.zeros_like(distribution) + 1. / distribution.size)
        dist_mean = np.mean(distribution)
        ax.set_title('Node '+str(indices[i]+1),fontsize=7,pad=0.005)
        means.append(dist_mean)
        ind = int((dist_mean-2.8)*(256)/(12-2.8)) # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        col_ar = cm.lipari.colors[ind]
        col_tuple = tuple(col_ar)
        ax.patch.set_facecolor(col_tuple)
        ax.patch.set_alpha(0.9)
        distributions.append(distribution)
        ax.tick_params(axis='both',labelsize=6.5)
        ax.set_yticks(np.arange(0,0.4,0.1))
        ax.set_xticks([0,5,10,15])
        ax.tick_params(axis='both', which='major', pad=0.2,width=0.5,length=1.9)

        # if dist_mean < 10:
        #     ax.text(0.1,0.33,('Mean='+str(round(dist_mean,1))+'m/s'),fontsize=5)
        # else:
        #     ax.text(0.1,0.33,('Mean='+str(round(dist_mean,1))+'m/s'),fontsize=5)
        if i > (Nx*Ny) - Ny - 1 or i==26 or i==27:
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel('(m/s)',fontsize=7,labelpad=0.2)
        if i % 4 == 0:
            ax.set_ylabel('Frequency',fontsize=7)
        if indices[i] in bad_nodes:
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='grey', linewidth=1, alpha=0.7,
                    path_effects=[pe.Stroke(linewidth=2, foreground='k',alpha=0.7), pe.Normal()],zorder=21)
            ax.plot([0, 1], [1, 0], transform=ax.transAxes, color='grey', linewidth=1, alpha=0.7,
                    path_effects=[pe.Stroke(linewidth=2, foreground='k',alpha=0.7), pe.Normal()],zorder=21)

    im = comap.ScalarMappable(norm=Normalize(3,11),cmap='cmc.lipari')
    cbar = fig.colorbar(im,ax=axes,fraction=0.046, pad=0.04,shrink=0.6,location='right')
    cbar.set_label('Mean Wind Speed (m/s)',size=8)
    cbar.ax.tick_params(labelsize=7)
    #plt.show()
    plt.savefig('plots/dist-'+seas+'-2x2-final.pdf',bbox_inches='tight')

    return None

def plot_distributions_14x2(distributions,bad_nodes):
    a=40
        
    fig, axes = plt.subplots(nrows=int(Nx/2), ncols=int(Ny*2+1), figsize=(3.75,6.2),sharex=True,sharey=True,
                             subplot_kw={'aspect':a},gridspec_kw = {'width_ratios':[0.5,0.5,0.15,0.5,0.5]})
    vmin = np.nanmin(obs_train.Wind.values)
    vmax = np.nanmax(obs_train.Wind.values)
    means = []
    
    for i, ax in enumerate(axes.flatten()):
        if (i-2) % 5 == 0:
            ax.axis('off')
            continue
        elif i > 37:
            ax.axis('off')
            continue
        elif i > 2:
            i -= ((i-3) // 5)+1
        distribution = distributions[indices[i]]  # wind obs that belong to this node -> this is our distribution
        ax.hist(distribution, range=(vmin,vmax),color='black',
                weights=np.zeros_like(distribution) + 1. / distribution.size)
        dist_mean = np.mean(distribution)
        means.append(dist_mean)
        ax.set_title('Node '+str(indices[i]+1),fontsize=7,pad=0.1)
        ind = int((dist_mean-2.8)*(256)/(12-2.8)) # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        col_ar = cm.lipari.colors[ind]
        col_tuple = tuple(col_ar)
        ax.patch.set_facecolor(col_tuple)
        ax.patch.set_alpha(0.9)
        distributions.append(distribution)
        ax.tick_params(axis='both',labelsize=6.5,pad=0.2,width=0.5,length=2)
        ax.set_yticks(np.arange(0,0.4,0.1))
        ax.set_xticks([0,5,10,15])
        # if dist_mean < 10:
        #     ax.text(0.1,0.33,('Mean='+str(round(dist_mean,1))+'m/s'),fontsize=5)
        # else:
        #     ax.text(0.1,0.33,('Mean='+str(round(dist_mean,1))+'m/s'),fontsize=5)
        if i > 23:
            ax.set_xlabel('(m/s)',fontsize=7,labelpad=0.2)
        if i % 4 == 0:
            ax.set_ylabel('Frequency',fontsize=7)
        if indices[i] in bad_nodes:
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='grey', linewidth=1.5, alpha=0.7,
                    path_effects=[pe.Stroke(linewidth=2.5, foreground='k',alpha=0.7), pe.Normal()],zorder=21)
            ax.plot([0, 1], [1, 0], transform=ax.transAxes, color='grey', linewidth=1.5, alpha=0.7,
                    path_effects=[pe.Stroke(linewidth=2.5, foreground='k',alpha=0.7), pe.Normal()],zorder=21)

    im = comap.ScalarMappable(norm=Normalize(3,11),cmap='cmc.lipari')

    cbar = fig.colorbar(im,ax=axes,fraction=0.046, pad=0.05,shrink=0.6,location='right')
    cbar.set_label('Mean Wind Speed (m/s)',size=8)
    cbar.ax.tick_params(labelsize=7)
    #plt.show()
    plt.savefig('plots/dist-'+seas+'-2x2-final.pdf',bbox_inches='tight')

    return None

def plot_distributions_6x4(distributions,bad_nodes):
        
    fig, axes = plt.subplots(nrows=Nx, ncols=Ny, figsize=(Ny*1.6,Nx*1.64),sharex=True,sharey=True,
                             layout='constrained',subplot_kw={'aspect':48},gridspec_kw = {'wspace':0.03, 'hspace':0.02})
    vmin = np.nanmin(obs_train.Wind.values)
    vmax = np.nanmax(obs_train.Wind.values)
    means = []
    
    for i, ax in enumerate(axes.flatten()):
        distribution = distributions[indices[i]]  # wind obs that belong to this node -> this is our distribution
        ax.hist(distribution, range=(vmin,vmax),color='black',
                weights=np.zeros_like(distribution) + 1. / distribution.size)
        dist_mean = np.mean(distribution)
        means.append(dist_mean)
        ind = int((dist_mean-2.8)*(256)/(12-2.8)) # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        col_ar = cm.lipari.colors[ind]
        col_tuple = tuple(col_ar)
        ax.patch.set_facecolor(col_tuple)
        ax.patch.set_alpha(0.9)
        distributions.append(distribution)
        ax.tick_params(axis='both',labelsize=12)
        ax.set_xticks([0,5,10,15])
        ax.set_title('Node '+str(indices[i]+1),fontsize=12,pad=0.06)
        # if dist_mean < 10:
        #     ax.text(3,0.36,('Mean = '+str(round(dist_mean,1))+'m/s'),fontsize=9)
        # else:
        #     ax.text(2,0.36,('Mean = '+str(round(dist_mean,1))+'m/s'),fontsize=9)
        if i > (Nx*Ny) - Ny - 1:
            ax.set_xlabel('(m/s)',fontsize=13)
        if i % Ny == 0:
            ax.set_ylabel('Frequency',fontsize=13)

        if i in bad_nodes:
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='grey', linewidth=3, alpha=0.7,
                    path_effects=[pe.Stroke(linewidth=4, foreground='k',alpha=0.7), pe.Normal()],zorder=21)
            ax.plot([0, 1], [1, 0], transform=ax.transAxes, color='grey', linewidth=3, alpha=0.7,
                    path_effects=[pe.Stroke(linewidth=4, foreground='k',alpha=0.7), pe.Normal()],zorder=21)
        if indices[i] in bad_nodes:
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='grey', linewidth=3, alpha=0.7,
                    path_effects=[pe.Stroke(linewidth=4, foreground='k',alpha=0.7), pe.Normal()],zorder=21)
            ax.plot([0, 1], [1, 0], transform=ax.transAxes, color='grey', linewidth=3, alpha=0.7,
                    path_effects=[pe.Stroke(linewidth=4, foreground='k',alpha=0.7), pe.Normal()],zorder=21)

    im = comap.ScalarMappable(norm=Normalize(3,11),cmap='cmc.lipari')

    cbar = fig.colorbar(im,ax=axes,fraction=0.046, pad=0.05,shrink=0.6,location='right')
    cbar.set_label('Mean Wind Speed (m/s)',size=14)
    cbar.ax.tick_params(labelsize=12)
    #plt.show()
    plt.savefig('plots/dist-'+seas+'-2x2-final.pdf',bbox_inches='tight')

    return None


def plot_domain():
    proj=ccrs.PlateCarree()
    fig, ax = plt.subplots(nrows=1, ncols=1,layout='constrained',figsize=(10,10),
                             subplot_kw={'projection': proj, 'aspect':1.7},gridspec_kw = {'wspace':0.1, 'hspace':0.08})
    sizes_lat = [14,18,22,26]
    sizes_lon = [24,32,40,48]
    central_lat =55
    central_lon =240

    ax.set_extent(([202,305,38,67]))
    
    ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
        name='admin_0_boundary_lines_land', scale='50m', facecolor='none', edgecolor='k',alpha=0.6,linewidth=0.7))
    ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', 
        name='admin_1_states_provinces_lines', scale='50m',facecolor='none', edgecolor='k',alpha=0.6,linewidth=0.7))
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', \
        scale='50m', edgecolor='k', facecolor='none', alpha=0.6,linewidth=0.7))
    for s in range(len(sizes_lat)):
        lat_min = central_lat - sizes_lat[s] / 2
        lat_max = central_lat + sizes_lat[s] / 2
        lon_min = central_lon - sizes_lon[s] / 2
        lon_max = central_lon + sizes_lon[s] / 2
        lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
        lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
        ax.plot(lons, lats, transform=ccrs.PlateCarree())
    #ax.scatter(360-120.4306,55.6986,c='yellow',transform=ccrs.PlateCarree(),s=5,marker='*',zorder=20)

    # Create gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.8, color='black', alpha=0.2,linestyle='--')
    # Manipulate gridlines number and spaces
    gl.ylocator = mticker.FixedLocator(np.arange(-90,90,10))
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10))
    gl.xlabel_style = {'size': 14,'rotation':40}
    gl.ylabel_style = {'size': 14}
    gl.bottom_labels = True
    gl.left_labels = True

    #plt.suptitle('z500 clusters')
    plt.savefig('plots/domains.pdf',bbox_inches='tight')


if __name__ == '__main__':
    # should redo DJF 24h plots and table.
    # and redo MAM everything because changed error metric.
    # for others, have done the initial 'forecast' thing but no plots or stats so far.
    res = 24 # time resolution in hours
    res_str = str(res)+'h'
    train_period=slice("2009-10-01","2020-09-23")
    val_period = slice("2020-10-01","2022-03-31")
    test_period_gefs = slice("2022-03-25","2024-03-25")  # 6 days earlier so that dates line up
    test_period = slice("2022-04-01","2024-04-01")
    #test_period_gefs = slice("2020-09-24","2022-03-24")
    #test_period = slice("2020-10-01","2022-03-31")
    seas = "DJF"
    if seas == "DJF" or seas == "JJA":
        level = 1000
        agg = 2
        Nx = 15
        Ny = 2
    elif seas == "SON":
        level = 850
        agg = 1
        Nx = 14
        Ny = 2
    elif seas == "MAM":
        level = 850
        agg = 2
        Nx = 6
        Ny = 4
    lat_offset = 9
    lon_offset = 16

    do = False
    num_members = 15

    # setup
    N_nodes = Nx * Ny
    title = '24h-'+seas+'-'+str(level)+'-final'

    lat = np.arange(55-lat_offset,55+lat_offset+0.5,0.5)[::-1]
    lon = np.arange(240-lon_offset,240+lon_offset+0.5,0.5)
    colors = ["#0072B2","#CC79A7","#BFBFBF","#4268B3"]

    # this is for lead time. if res > 6H, t_step is slice so that we pick up multiple z forecasts
    t_step = []
    if res <= 24:
        for d in range(6,15): # each day in week 2
            if res == 24:
                t_step.append(slice(np.array(int((d*24)*1e9*60*60),dtype='timedelta64[ns]'), 
                                    np.array(int(((d+0.9)*24)*1e9*60*60),dtype='timedelta64[ns]')))
            elif res == 12:
                t_step.append(slice(np.array(int((d*24)*1e9*60*60),dtype='timedelta64[ns]'), 
                                    np.array(int(((d+0.48)*24)*1e9*60*60),dtype='timedelta64[ns]')))
                t_step.append(slice(np.array(int(((d+0.5)*24)*1e9*60*60),dtype='timedelta64[ns]'),  
                                    np.array(int(((d+0.9)*24)*1e9*60*60),dtype='timedelta64[ns]')))
            elif res == 6:
                t_step.append(np.array(int(d*24*1e9*60*60),dtype='timedelta64[ns]'))
                t_step.append(np.array(int((d*24+6)*1e9*60*60),dtype='timedelta64[ns]'))
                t_step.append(np.array(int((d*24+12)*1e9*60*60),dtype='timedelta64[ns]'))
                t_step.append(np.array(int((d*24+18)*1e9*60*60),dtype='timedelta64[ns]'))
            else:
                print('only configured for 6h, 12h, and 24h right now')
                raise ValueError
    elif res == 48:
        t_step.append(slice(np.array(int((7*24)*1e9*60*60),dtype='timedelta64[ns]'),  # day 7,8
                                np.array(int(((8+0.9)*24)*1e9*60*60),dtype='timedelta64[ns]')))
        t_step.append(slice(np.array(int((9*24)*1e9*60*60),dtype='timedelta64[ns]'),  # day 9,10
                                np.array(int(((10+0.9)*24)*1e9*60*60),dtype='timedelta64[ns]')))
        t_step.append(slice(np.array(int((11*24)*1e9*60*60),dtype='timedelta64[ns]'),  # day 11,12
                                np.array(int(((12+0.9)*24)*1e9*60*60),dtype='timedelta64[ns]')))
        t_step.append(slice(np.array(int((13*24)*1e9*60*60),dtype='timedelta64[ns]'),  # day 13,14
                                np.array(int(((14+0.9)*24)*1e9*60*60),dtype='timedelta64[ns]')))
    elif res == 168:
        t_step.append(slice(np.array(int((7*24)*1e9*60*60),dtype='timedelta64[ns]'),  # day 7-14
                                np.array(int(((14+0.9)*24)*1e9*60*60),dtype='timedelta64[ns]')))
    else:
        print('Invalid resolution')
        raise ValueError
    
    t_step_plot = []
    if res > 6:
        for l in t_step:
            t_step_plot.append(l.start)
    else:
        t_step_plot = t_step

    # obs data
    print('Prepping data...')
    obs_full = xr.open_dataset('~/Nextcloud/thesis/bm_cleaned_all.nc')
    obs_full = resample_mean(obs_full,'obs',res)
    obs_full = obs_full.sel(index=obs_full.index.dt.season==seas)

    # use training obs for climatology, val obs for finding bad nodes, testing obs for testing
    obs_train = obs_full.sel(index=train_period)
    clim_prob, clim_det = get_climo_det(obs_train)  # for both datasets need to select the intended hour
    obs_val = obs_full.sel(index=val_period)
    obs_val = obs_val.dropna(dim='index')
    obs_test = obs_full.sel(index=test_period)
    obs_full = None  # free up memory

   # era data (used for validation - finding bad nodes)
    if level == 850:
        era = xr.open_dataset('era-850-2009-2022.nc').sel(latitude=lat,longitude=lon-360)
    else:
        era = xr.open_dataset('era-2009-2022-a.nc').sel(latitude=lat,longitude=lon-360,level=level)
    erav = era.sel(time=val_period)
    era = era.sel(time=train_period)
    print('Processing data...')
    # taking out all (spatial and temporal) anomaly
    clim = era.groupby("time.dayofyear").mean(dim=["time"])  # clim for each doy for each pixel
    clim_concat = xr.concat([clim.isel(dayofyear=slice(330,366)),clim,clim.isel(dayofyear=slice(0,36))],dim='dayofyear')
    cutoff=0.03
    b, a = signal.butter(5, cutoff, btype='lowpass')
    dUfilt = signal.filtfilt(b, a, clim_concat.gh.values,axis=0)
    dUfilt = dUfilt[36:-36,:,:]
    clim.gh.values = dUfilt

    erav = erav.groupby("time.dayofyear") - clim
    erav = resample_mean(erav,'era',6)
    erav = erav.sel(time=obs_val.index.values)  # only keep indices with valid obs

    # open som and dists
    print('Opening map and doing forecast...')
    with open('trained-map-'+title[4:]+'.pkl','rb') as handle:
        som = pickle.load(handle)
    with open('distributions-'+title[4:]+'.pkl','rb') as handle:
        distributions = pickle.load(handle)
    
    bad_nodes = era_stats(agg)

    indices = np.arange(N_nodes).reshape(Nx,Ny).T.flatten()
    if Nx < 8:
        indices=np.arange(N_nodes).reshape(Ny,Nx).flatten()
        #plot_som_6x4(Nx, Ny, som.z_raw, indices,bad_nodes)
        plot_distributions_6x4(distributions,bad_nodes)
    elif Nx == 15:
        indices = np.concatenate([np.arange(16).reshape(8,2),np.arange(16,32).reshape(8,2)],axis=1).flatten()
        #plot_som(Nx, Ny, som.z_raw, indices,bad_nodes)
        plot_distributions(distributions,bad_nodes)
    elif Nx == 14:
        indices = np.concatenate([np.arange(14).reshape(7,2),np.arange(14,28).reshape(7,2)],axis=1).flatten()
        #plot_som_14x2(Nx, Ny, som.z_raw, indices,bad_nodes)
        plot_distributions_14x2(distributions,bad_nodes)

    print('Done')


    
