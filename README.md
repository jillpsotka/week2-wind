# week2-wind
Masters thesis research!!

Improving week-2 hub height wind forecasts using upper-level nwp outputs + self-organizing maps to extrapolate down to hub height. Thesis: (insert link to thesis once published)

## 1. Required data
- ERA5 geopotential at desired levels (500,700,850,1000 hPa) for training & validation period
- hub-height wind observations for training, validation, and testing period (WFRT members: https://gl.tawhiri.eos.ubc.ca/emwxnet/recipes/-/tree/master/get_obs)
qc_obs.py
- GEFS or any other forecast of geopotential heights for testing period
get-gefs.py
gefs_merge.py

## 2. Optimizing SOM hyperparameters
do_som_optimization.py
- for each season, test all the different combos of SOM parameters (takes 12+ hours)
- creates statistics files for plot_stats.py to open, read, and investigate
(not completely automated, need to play in plot_stats.py to choose best SOM)

## 3. Forecasting
- put chosen SOM parameters into do_som.py and run it to train and save the final SOM
- plotting-som.py makes plots of SOM and distributions (will have to update this script if making SOMs that are different sizes than in my thesis figures)
- plotting.py does forecasting for the testing period, can make plots of statistics
- plotting-summary.py to make summary figures
