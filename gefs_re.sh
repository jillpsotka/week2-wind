#!/bin/bash
# bash to download gefs reanalysis

cd /users/jpsotka/repos/week2-wind/data
YEAR=$1


for MONTH in 0{1..9} {10..12} ; do

	for DAY in 0{1..9} {10..31} ; do
		HOST="ftp://ftp.emc.ncep.noaa.gov/GEFSv12/reanalysis/FV3_reanalysis/$YEAR/$MONTH/$DAY/"

		wget -nv -m $HOST

		python /users/jpsotka/repos/week2-wind/gefs_grib.py $YEAR $MONTH $DAY # takes the files we just downloaded, cuts them, saves as netcdf, and deletes the originals

	done

	python /users/jpsotka/repos/week2-wind/gefs_merge.py $YEAR $MONTH

done
