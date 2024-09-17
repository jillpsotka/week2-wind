#!/bin/bash
source /users/jpsotka/miniconda3/bin/activate thesis
cd /users/jpsotka/repos/week2-wind

for MEM in {18..20} ; do
    python get-gefs.py $MEM
    python gefs_merge.py $MEM
done
