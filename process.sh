#!/bin/bash
find ~/projects/rpp-bengioy/jpcohen/icentia-mila-research -name "*npz" ! -name "*batched*" -exec echo {} \; -exec python data_split.py {} ~/projects/rpp-bengioy/jpcohen/icentia12k/ \;
