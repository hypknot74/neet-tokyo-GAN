# LMDB_PATH='../dataset/lmdb_1024'
# LMDB_PATH='./dataset/lmdb/cropped_dataset/muct'
LMDB_PATH='./dataset/lmdb/nt/'

SAMPLE_DIR='base'
CHECKPOINT_DIR='base'

BATCH_SIZE=16
ITER=50000
START_ITER=0

N_GPU=1
WANDB=False


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


cd ..


get_ipython().getoutput("python train.py --n_gpu $N_GPU --conf config/config-t.jsonnet \")
training.batch=$BATCH_SIZE training.iter=$ITER training.start_iter=$START_ITER \
path=$LMDB_PATH wandb=$WANDB sample_dir=$SAMPLE_DIR checkpoint_dir=$CHECKPOINT_DIR


LMDB_PATH='dataset/lmdb/cropped_dataset/tezuka'
SAMPLE_DIR='tezuka_test'
CHECKPOINT_DIR='tezuka_test'

CKPT='./checkpoint/tezuka_test/020000.pt'

BATCH_SIZE=16
ITER=150000
START_ITER=20000

N_GPU=1
WANDB=True


get_ipython().getoutput("python train.py --n_gpu $N_GPU --ckpt $CKPT --conf config/config-t.jsonnet \")
training.batch=$BATCH_SIZE training.iter=$ITER training.start_iter=$START_ITER \
path=$LMDB_PATH wandb=$WANDB sample_dir=$SAMPLE_DIR checkpoint_dir=$CHECKPOINT_DIR \
training.ckpt=$CKPT



