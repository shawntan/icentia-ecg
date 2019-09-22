#!/bin/bash

BASEPATH=/lustre03/project/6008064/jpcohen/icentia-mila-research
TARGETPATH=/lustre03/project/6008064/jpcohen/icentia12k

for i in {0..12000}; do
    echo $i;
    python data_split.py $BASEPATH/$i.npz $TARGETPATH/
    sleep 1
done

# old method
#find ~/projects/rpp-bengioy/jpcohen/icentia-mila-research -name "*npz" ! -name "*batched*" -exec echo {} \; -exec python data_split.py {} ~/projects/rpp-bengioy/jpcohen/icentia12k/ \;
