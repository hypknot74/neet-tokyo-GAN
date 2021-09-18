PATH='../../dataset/lmdb_1024'

ITER=50000
SIZE=1024


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


get_ipython().getoutput("python ../train.py --iter $ITER --size $SIZE --wandb $PATH")



