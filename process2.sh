#!/bin/bash

open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}
run_with_lock(){
    local x
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    printf '%.3d' $? >&3
    )&
}

N=16
open_sem $N
toprocess=`find ~/projects/rpp-bengioy/jpcohen/icentia-mila-research -name "*npz" ! -name "*batched*" -print`
for thing in $toprocess; do
    run_with_lock bash process-single.sh $thing
done 
sleep 2000
#find ~/projects/rpp-bengioy/jpcohen/icentia-mila-research -name "*npz" ! -name "*batched*" -exec echo {} \; -exec python data_split.py {} ~/projects/rpp-bengioy/jpcohen/icentia12k/ \;
