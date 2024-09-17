from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib as mpl
from cmcrameri import cm
import re
import ast
import csv
import warnings
warnings.filterwarnings("ignore")


# finding ideal map configuration


mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12

def parse_list(x):
    return ast.literal_eval(x)

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
    ds['Nnodes'] = pd.to_numeric(ds['Nnodes'])
    ds = ds.sort_values(['Nnodes','Nx'])
    return ds


def QE_TE_plot(ds):
    ds = sort_by_N(ds)
    fig, ax = plt.subplots(1)
    ax.plot(ds['N_nodes'].values,ds.QE.values,c='blue',label='QE')
    ax.set_ylabel('Quantization error')
    ax2 = ax.twinx()
    ax2.scatter(ds['Nnodes'].values,ds.TE.values,color='black')
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
    colors = cmap(np.linspace(0.08, 1, 10))
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


def domain_stats(ds):
    small = ds[ds['lat'] < 8]
    med = ds[ds['lat'] == 9]
    large = ds[ds['lat'] > 10]

    domains = [small,med,large]
    tits = ['small','med','large']
    stats = ['R','crpss_avg','D','rmse']

    fig, axes = plt.subplots(nrows=2, ncols=2)
    for i, ax in enumerate(axes.flatten()):
        for k in range(len(domains)):
            ax.scatter(domains[k]['N_nodes'],domains[k][stats[i]],label=tits[k])
        ax.set_ylabel(stats[i])
    plt.legend()

    plt.show()
    

def domain_pf(ds_f):
    fig, axes = plt.subplots(nrows=2, ncols=2)

    for i, ax in enumerate(axes.flatten()):  # each season
        ds = ds_f[i]
        #ds = ds[ds['Ny']>1]
        small = ds[ds['lat'] < 8]
        med = ds[ds['lat'] == 9]
        large = ds[ds['lat'] > 10]

        domains = [small,med,large]
        tits = ['small','med','large']

    
        for k in range(len(domains)):  # 3 colour scatter on each subplot
            ax.scatter(domains[k]['N_nodes'],domains[k]['PF'],label=tits[k])
        ax.set_ylabel('PF')
        ax.set_title(titles[i])
    plt.legend()

    plt.show()
        



def heat_map(ds,var,tit):
    # how good at sorting into nodes -> topographic error
    # how different are the wind dists -> k-s significance
    # how well does it predict -> crpss

    # create some strings for labels
    labels = []
    for l in ds.index:
        labels.append(str(ds['Nx'][l])+'x'+str(ds['Ny'][l]))
    ds['labels'] = labels


    cmap = plt.cm.hot_r
    colors = cmap(np.linspace(0.08, 1, 14))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", colors)   

    sc = plt.scatter(ds['N_nodes'],ds['PF'],c=ds[var],cmap=cmap)
    for i, txt in enumerate(ds['labels']):
        plt.annotate(txt, (ds['N_nodes'][i], ds['PF'][i]),fontsize=6)
    cbar = plt.colorbar(sc)
    cbar.set_label(var)
    plt.xlabel('N nodes')
    #plt.gca().invert_xaxis()
    plt.ylabel('Pseudo-F')
    plt.title(tit+', 50kPa, 24-h, reanalysis')
    plt.tight_layout()
    plt.savefig('plots/heat-map-'+var+'-'+tit+'.png',dpi=200)
    plt.close()


    return None


def compare_levels(ds,season=''):
    # ds is len-3 list of pd arrays (one for each level)
    fig, axes = plt.subplots(nrows=2, ncols=2,dpi=200)
    metrics = ['PF','R-gefs','crpss_avg','bss-50-gefs']
    small = ds[0]
    med = ds[1]
    large = ds[2]

    domains = [small,med,large]
    tits = ['500','700','1000']

    for i, ax in enumerate(axes.flatten()):  # each metric

        for k in range(len(domains)):  # 3 colour scatter on each subplot
            ax.scatter(domains[k]['N_nodes'],domains[k][metrics[i]],label=tits[k])
        ax.set_title(metrics[i])
        ax.set_xlabel('# nodes')
    plt.legend()
    plt.suptitle(season)
    plt.tight_layout()

    plt.savefig('plots/levels-stats-'+season+'.png')


    return None


def get_best(ds):
    # find the best performing maps at each lead time
    crpss_avg = []
    crpss_6 = []
    crpss_7 = []
    crpss_8 = []
    crpss_9 = []
    crpss_10 = []
    crpss_11 = []
    crpss_12 = []
    crpss_13 = []
    crpss_14 = []
    for crpss in ds['CRPSS']:
        if crpss:
            crpss_6.append(crpss[0])
            crpss_7.append(crpss[1])
            crpss_8.append(crpss[2])
            crpss_9.append(crpss[3])
            crpss_10.append(crpss[4])
            crpss_11.append(crpss[5])
            crpss_12.append(crpss[6])
            crpss_13.append(crpss[7])
            crpss_14.append(crpss[8])
            crps_ls = []
            for c in crpss:
                crps_ls.append(c)
            crpss_avg.append(np.mean(crps_ls))
        else:
            crpss_avg.append(np.nan)
    ds['crpss_avg'] = crpss_avg
    ds['crpss_6'] = crpss_6
    ds['crpss_7'] = crpss_7
    ds['crpss_8'] = crpss_8
    ds['crpss_10'] = crpss_10
    ds['crpss_12'] = crpss_12
    ds['crpss_14'] = crpss_14

    ds = ds.sort_values(['crpss_avg'],na_position='first')
    ds = ds.reset_index()
    #ds=ds.drop(range(int(0.5*ds.shape[0])))  # drop worst half

    best_avg = ds.loc[ds['crpss_avg'].idxmax()]
    best_6 = ds.loc[ds['crpss_6'].idxmax()]
    best_8 = ds.loc[ds['crpss_8'].idxmax()]
    best_10 = ds.loc[ds['crpss_10'].idxmax()]
    best_12 = ds.loc[ds['crpss_12'].idxmax()]
    best_14 = ds.loc[ds['crpss_14'].idxmax()]

    # plt.plot(best_avg['CRPSS'],label='avg')
    # plt.plot(best_6['CRPSS'],label='6')
    # plt.plot(best_8['CRPSS'],label='8')
    # plt.plot(best_10['CRPSS'],label='10')
    # plt.plot(best_12['CRPSS'],label='12')
    # plt.plot(best_14['CRPSS'],label='14')
    # plt.legend()

    return ds


def rank_all(ds1):
    # rank the different map configurations to find the best one for each season

    # ds = ds1.sort_values(['TE'],na_position='first',ascending=False)
    # ds = ds.reset_index(drop=True)
    # ds=ds.drop(range(int(0.25*ds.shape[0])))


    ds1['score'] = 0
    ds = ds1.sort_values(['PF-gefs'],na_position='first')
    ds = ds.reset_index(drop=True)
    ds['score'] = ds['score'] + ds.index

    ds = ds.sort_values(['crpss_avg','PF'],na_position='first')
    ds = ds.reset_index(drop=True)
    ds['score'] = ds['score'] + ds.index

    #ds = ds.sort_values(['score'])

    # important: plots like this help find best SOM
    # I tried to ignore outliers that had rly good crpss to avoid overtraining to validation data
    # plt.scatter(ds[ds['level']=='700']['PF'],ds[ds['level']=='700']['crpss_avg'],color='orange',label='700')
    # plt.scatter(ds[ds['level']=='500']['PF'],ds[ds['level']=='500']['crpss_avg'],color='blue',label='500')
    # plt.scatter(ds[ds['level']=='850']['PF'],ds[ds['level']=='850']['crpss_avg'],color='green',label='850')
    # plt.scatter(ds[ds['level']=='1000']['PF'],ds[ds['level']=='1000']['crpss_avg'],color='black',label='1000')
    # plt.show()
    return ds


def add_quotes_to_nested_lists(line):
    # Regular expression to match nested lists
    pattern = r'\[\[.*?\]\]|\[.*?\]'
    
    # Find all nested lists in the line
    nested_lists = re.findall(pattern, line)
    
    # Replace each nested list with its quoted version
    for sublist in nested_lists:
        line = line.replace(sublist, '"' + sublist + '"')
    
    return line

def mems():
    season = 'DJF'
    s1000_0 = 'stats-24h-'+season+'-1000-15members-0.txt'
    s1000_1 = 'stats-24h-'+season+'-1000-15members-1.txt'
    s1000_2= 'stats-24h-'+season+'-1000-15members-2.txt'

    s850_0 = 'stats-24h-'+season+'-850-15members-0.txt'
    s850_1 = 'stats-24h-'+season+'-850-15members-1.txt'
    s850_2= 'stats-24h-'+season+'-850-15members-2.txt'

    s700_0 = 'stats-24h-'+season+'-700-15members-0.txt'
    s700_1 = 'stats-24h-'+season+'-700-15members-1.txt'
    s700_2= 'stats-24h-'+season+'-700-15members-2.txt'

    s500_0 = 'stats-24h-'+season+'-500-15members-0.txt'
    s500_1 = 'stats-24h-'+season+'-500-15members-1.txt'
    s500_2= 'stats-24h-'+season+'-500-15members-2.txt'

    # s1000_0test15 = 'stats-24h-'+season+'-1000-15members-0.txt'
    # s1000_1test15 = 'stats-24h-'+season+'-1000-15members-1.txt'
    # s1000_2test15= 'stats-24h-'+season+'-1000-15members-2.txt'

    # s1000_0test10 = 'stats-24h-'+season+'-1000-10members-0.txt'
    # s1000_1test10 = 'stats-24h-'+season+'-1000-10members-1.txt'
    # s1000_2test10= 'stats-24h-'+season+'-1000-10members-2.txt'

    # s1000_0test20 = 'stats-24h-'+season+'-1000-20members-0.txt'
    # s1000_1test20 = 'stats-24h-'+season+'-1000-20members-1.txt'
    # s1000_2test20= 'stats-24h-'+season+'-1000-20members-2.txt'

    # s850_0test = 'stats-24h-'+season+'-850-test-0.txt'
    # s850_1test = 'stats-24h-'+season+'-850-test-1.txt'
    # s850_2test= 'stats-24h-'+season+'-850-test-2.txt'

    # s700_0test = 'stats-24h-'+season+'-700-test-0.txt'
    # s700_1test = 'stats-24h-'+season+'-700-test-1.txt'
    # s700_2test= 'stats-24h-'+season+'-700-test-2.txt'


    # titles = [season+' 1000 0 15',season+' 1000 1 15',season+' 1000 2 15',
    #           season+' 1000 0 10',season+' 1000 1 10',season+' 1000 2 10',
    #           season+' 1000 0 20',season+' 1000 1 20',season+' 1000 2 20']
    # files = [s1000_0test15,s1000_1test15,s1000_2test15,
    #          s1000_0test10,s1000_1test10,s1000_2test10,
    #          s1000_0test20,s1000_1test20,s1000_2test20]

    titles = [season+' 1000 0',season+' 1000 1',season+' 1000 2',
              season+' 850 0',season+' 850 1',season+' 850 2',
              season+' 700 0',season+' 700 1',season+' 700 2',
              season+' 500 0',season+' 500 1',season+' 500 2']
    files = [s1000_0,s1000_1,s1000_2,s850_0,s850_1,s850_2,s700_0,s700_1,s700_2,s500_0,s500_1,s500_2]

    indv_ds = []

    # do some stuff so that csv reader keeps the lists
    # https://stackoverflow.com/questions/67382217/how-to-read-a-csv-dialect-containing-a-list-in-square-brackets
    for f in range(len(files)):
        filename = files[f]
        title = titles[f]
        level = title.split()[1]
        aggression = title.split()[2]
        #mems = title.split()[3]

        data = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for line in reader:
                # Split the line by commas, preserving the structure of nested lists
                parts = []
                open_bracket = False
                open_double = False
                current_part = ''
                for char in ','.join(line):
                    if char == ',' and not open_bracket:
                        parts.append(current_part)
                        current_part = ''
                    else:
                        current_part += char
                        if char == '[':
                            open_bracket = True
                            if len(current_part) > 1 and (current_part[-2] == '[' or current_part[-2] == ' '):
                                open_double = True
                        elif char == ']':
                            if open_double:
                                if current_part[-2] == ']' or current_part[-2] == 'n':
                                    open_double = False
                                    open_bracket = False
                            else:
                                open_bracket = False
                parts.append(current_part)
                data.append(parts)

        stats = pd.DataFrame(data[1:-1], columns=data[0])

        stats['level'] = level
        stats['aggression'] = aggression
        #stats['mems'] = mems

        stats = sort_by_N(stats)

        list_str = ['bss-gefs','D-gefs', 'CRPSS','frac-discarded','PF-gefs','KS-gefs','bias-gefs','R-gefs','mae-gefs']

        for l in stats.index:
            if not stats['mae-gefs'][l]:  # this map had all bad nodes
                stats=stats.drop(l)
                continue
            if stats['CRPSS'][l][0] == '-':
                stats=stats.drop(l)
                continue
            for s in list_str:
                stats[s][l] = stats[s][l].replace('nan', 'np.nan')
                stats[s][l] = eval(stats[s][l])
            if stats['frac-discarded'][l][0] > 0.5 or stats['frac-discarded'][l][1] > 0.5 or stats['frac-discarded'][l][2] > 0.5:
                stats=stats.drop(l)
                continue

        indv_ds.append(stats)

        #heat_map(stats,'R-gefs',title)
        #domain_stats(stats)

    stats = indv_ds[0]
    for x in indv_ds[1:]:
        stats = stats.append(x)
    stats['PF'] = stats.PF.astype(float)
    stats['TE'] = stats.TE.astype(float)
    stats['QE'] = stats.QE.astype(float)
    stats = get_best(stats)
    stats = rank_all(stats)

    #compare_levels(indv_ds,season)

    print('done')


if __name__ ==  "__main__":
    mems()
    summer = 'stats-24h-anomalies-som-500-summer.txt'
    #smems = 'stats-24h-anomalies-som-500-members.txt'
    #fall ='stats-24h-anomalies-som-500-fall.txt'
    #winter = 'stats-24h-anomalies-som-500-winter.txt'
    #spring = 'stats-24h-anomalies-som-500-spring.txt'


    titles = ['summer']
    files = [summer]

    indv_ds = []

    # do some stuff so that csv reader keeps the lists
    # https://stackoverflow.com/questions/67382217/how-to-read-a-csv-dialect-containing-a-list-in-square-brackets
    for f in range(len(files)):
        filename = files[f]
        title = titles[f]
        converted = StringIO()
        with open(filename) as file:
            converted.write(file.read().replace('[', ']'))
        converted.seek(0)
        stats = pd.read_csv(converted,sep=',',header=0,skipinitialspace=True,quotechar=']')

        stats = sort_by_N(stats)

        crpss_avg = []
        crpss_avg_gefs=[]
        for l in stats.index:
            # stats['CRPSS'][l] = stats['CRPSS'][l].replace('nan', 'np.nan')
            # stats['CRPSS'][l] = stats['CRPSS'][l].replace('\n', '')
            # stats['CRPSS'][l] = stats['CRPSS'][l].replace(' ', ',')
            # stats['CRPSS'][l] = re.sub(r',+', ',', stats['CRPSS'][l])


            # crpss = np.array(eval(stats['CRPSS'][l]))
            # stats['CRPSS'][l] = crpss
            # crpss_avg.append(np.nanmean(crpss))
            stats['crpss-nodes'][l] = stats['crpss-nodes'][l].replace('nan', 'np.nan')
            crpss = np.array(eval(stats['crpss-nodes'][l]))
            stats['crpss-nodes'][l] = crpss
            crpss[crpss<0] = 0
            crpss_avg.append(np.nanmean(crpss))

            stats['crpss-nodes-gefs'][l] = stats['crpss-nodes-gefs'][l].replace('nan', 'np.nan')
            crpss = np.array(eval(stats['crpss-nodes-gefs'][l]))
            stats['crpss-nodes-gefs'][l] = crpss
            crpss[crpss<0] = 0
            crpss_avg_gefs.append(np.nanmean(crpss))

            stats['rmse-nodes-gefs'][l] = stats['rmse-nodes-gefs'][l].replace('nan', 'np.nan')
            crpss = np.array(eval(stats['rmse-nodes-gefs'][l]))
            stats['rmse-nodes-gefs'][l] = crpss
            stats['rmse-nodes'][l] = stats['rmse-nodes'][l].replace('nan', 'np.nan')
            crpss = np.array(eval(stats['rmse-nodes'][l]))
            stats['rmse-nodes'][l] = crpss

            stats['bias-nodes-gefs'][l] = stats['bias-nodes-gefs'][l].replace('nan', 'np.nan')
            crpss = np.array(eval(stats['bias-nodes-gefs'][l]))
            stats['bias-nodes-gefs'][l] = crpss
            stats['bias-nodes'][l] = stats['bias-nodes'][l].replace('nan', 'np.nan')
            crpss = np.array(eval(stats['bias-nodes'][l]))
            stats['bias-nodes'][l] = crpss

            stats['mae-nodes-gefs'][l] = stats['mae-nodes-gefs'][l].replace('nan', 'np.nan')
            crpss = np.array(eval(stats['mae-nodes-gefs'][l]))
            stats['mae-nodes-gefs'][l] = crpss
            stats['mae-nodes'][l] = stats['mae-nodes'][l].replace('nan', 'np.nan')
            crpss = np.array(eval(stats['mae-nodes'][l]))
            stats['mae-nodes'][l] = crpss

        stats['crpss_avg'] = crpss_avg
        #stats['crpss_avg-gefs'] = crpss_avg_gefs

        indv_ds.append(stats)

        heat_map(stats,'R',title)
        #domain_stats(stats)

    domain_pf(indv_ds)

    print('done')