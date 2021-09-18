PATH='../dataset/lmdb_512'

SAMPLE_DIR='512x512'
CHECKPOINT_DIR='512x512'

ITER=50000
SIZE=512


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


cd ..


get_ipython().getoutput("python train.py --iter $ITER --size $SIZE --wandb\")
--sample_dir $SAMPLE_DIR --checkpoint_dir $CHECKPOINT_DIR \
$PATH 



