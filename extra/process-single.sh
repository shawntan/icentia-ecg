BASEPATH=/lustre03/project/6008064/jpcohen/icentia-mila-research
TARGETPATH=/lustre03/project/6008064/jpcohen/icentia11k

if [[ ! -f "$BASEPATH/"$1".npz" ]]; then
    echo No source file
    exit
fi
if [[ ! -f "$TARGETPATH/"$1"_batched.pkl.gz" || ! -f "$TARGETPATH/"$1"_batched_lbls.pkl.gz" ]]; then
    echo creating $TARGETPATH/"$1"_batched.pkl.gz 
    timeout -s KILL 1000 \time -f 'time%e' python -u data_split.py $BASEPATH/$1.npz $TARGETPATH/
else
    echo exists $TARGETPATH/"$1"_batched.pkl.gz 
fi
