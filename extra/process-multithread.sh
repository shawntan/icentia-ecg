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

N=4
open_sem $N

for i in {0..12971}; do
    #echo $i;
    run_with_lock bash process-single.sh $i
    sleep 0.1
done
echo "Done"
sleep 200
echo "Done with sleep"
#find ~/projects/rpp-bengioy/jpcohen/icentia-mila-research -name "*npz" ! -name "*batched*" -exec echo {} \; -exec python data_split.py {} ~/projects/rpp-bengioy/jpcohen/icentia11k/ \;
