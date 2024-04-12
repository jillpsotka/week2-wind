from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib as mpl
from cmcrameri import cm


mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12


def plot_crpss(ds):
    crpss_max = []
    crpss_avg = []
    crpss_frac = []
    for l in ds.index:
        crpss = eval(stats['crpss-nodes'][l])
        crpss_max.append(np.max(crpss))
        crpss_avg.append(np.mean(crpss))
        crpss_frac.append(sum(x>0.05 for x in crpss) / len(crpss))

    plt.plot(crpss_avg)
    plt.show()


def sort_by_N(ds):
    ds['N_nodes'] = ds['Nx']*ds['Ny']
    ds = ds.sort_values(['N_nodes','Nx'])
    return ds


def QE_TE_plot(ds):
    ds = sort_by_N(ds)
    fig, ax = plt.subplots(1)
    ax.plot(ds['N_nodes'].values,ds.QE.values,c='blue',label='QE')
    ax.set_ylabel('Quantization error')
    ax2 = ax.twinx()
    ax2.scatter(ds['N_nodes'].values,ds.TE.values,color='black')
    ax2.set_ylabel('Topographic error')
    ax.set_xlabel('Number of nodes')
    ax2.spines['left'].set_color('blue')
    ax.yaxis.label.set_color('blue')

    plt.show()


def map_based_table(ds):
    # add 'count' column to count up points for being in the top of each statistic
    ds['Count'] = 0
    ds = ds[ds['Ny']>1]

    # first split ds into the 3 domain sizes
    small = ds[ds['lat'] < 8]
    med = ds[ds['lat'] == 9]
    large = ds[ds['lat'] > 10]

    QE_TE_plot(med)

    # get top-ranked configurations for each map-based statistic
    # maybe top 5 for now?
    num = 10
    for df in (small,med,large):
        df = df.sort_values('TE')  # topographic error, smaller is better
        df[:num]["Count"] += 1
        df = df.sort_values('QE')  # quantization error, smaller is better
        df[:num]["Count"] += 1
        df = df.sort_values('EV',ascending=False)  # explained variance, bigger is better
        df[:num]["Count"] += 1
        df = df.sort_values('PF',ascending=False)  # pseudo f, bigger is better
        df[:num]["Count"] += 1
        df = df.sort_values('KS-frac',ascending=False)  # fraction with k-s significance, bigger is better
        df[:num]["Count"] += 1

        # plot table

        # things are messed up - need to make an array that represents the table
        # one axis should be Nx which has labels 2 - 20
        # one should be Ny which has labels 1-6
        # values should be df.Count
        cells = np.zeros((len(np.unique(df.Ny)),len(np.unique(df.Nx))))
        x_offset = np.min(np.unique(df.Nx))
        y_offset = np.min(np.unique(df.Ny))

        for i, row in df.iterrows():
            x = row.Nx
            y = row.Ny
            cells[y-y_offset,x-x_offset] = row.Count

        norm = plt.Normalize(cells.min(), cells.max()+1)
        colours = plt.cm.hot_r(norm(cells))

        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])

        the_table=plt.table(cellText=cells,cellColours=colours, rowLabels=np.unique(df.Ny), colLabels=np.unique(df.Nx), 
                            loc='center')
        plt.show()

    return None


def shape_stats(ds):
    ds = ds[ds['lat'] > 10]
    cmap = plt.cm.hot_r
    colors = cmap(np.linspace(0.06, 1, 10))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", colors)   

    sc = plt.scatter(ds['Nx'],ds['Ny'],c=ds['bss-90-gefs'],cmap=cmap)
    cbar = plt.colorbar(sc)
    cbar.set_label('BSS (90th p)')
    plt.xticks(np.arange(2,21,2))
    plt.xlabel('Nx')
    plt.ylabel('Ny')
    plt.title('Stats for 50kPa, 24-h, +7day forecast')
    plt.tight_layout()
    plt.savefig('plots/bss90-gefs.png',dpi=200)


def heat_map(ds):
    # how good at sorting into nodes -> topographic error
    # how different are the wind dists -> k-s significance
    # how well does it predict -> crpss


    cmap = plt.cm.hot_r
    #colors = cmap(np.linspace(0, 0.8, 10))
    #cmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", colors)   

    sc = plt.scatter(ds['TE'],ds['PF'],c=ds['crpss_avg'],cmap=cmap)
    cbar = plt.colorbar(sc)
    cbar.set_label('Avg CRPSS')
    plt.xlabel('Topographic error')
    plt.gca().invert_xaxis()
    plt.ylabel('Pseudo-F')
    plt.title('Stats for 50kPa, 24-h, reanalysis')
    plt.tight_layout()
    plt.savefig('plots/heat-map.png',dpi=200)


    return None




if __name__ ==  "__main__":
    filename = 'stats-24h-anomalies-som-500-gefs.txt'

    # do some stuff so that csv reader keeps the lists
    # https://stackoverflow.com/questions/67382217/how-to-read-a-csv-dialect-containing-a-list-in-square-brackets
    converted = StringIO()
    with open(filename) as file:
        converted.write(file.read().replace('[', ']'))
    converted.seek(0)
    stats = pd.read_csv(converted,sep=',',header=0,skipinitialspace=True,quotechar=']')

    stats = sort_by_N(stats)

    crpss_avg = []
    for l in stats.index:
        stats['crpss-nodes'][l] = stats['crpss-nodes'][l].replace('nan', 'np.nan')
        crpss = np.array(eval(stats['crpss-nodes'][l]))
        crpss[crpss<0] = 0
        crpss_avg.append(np.nanmean(crpss))
    stats['crpss_avg'] = crpss_avg

    shape_stats(stats)

    print('done')