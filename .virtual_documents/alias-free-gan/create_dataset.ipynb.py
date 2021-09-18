MUCT_LMDB_PATH='dataset/lmdb/cropped_dataset/muct'
MUCT_DATASET_PATH='../Conditional-FastGAN/dataset/train_dataset/cropped_images/muct/'

TEZUKA_LMDB_PATH='dataset/lmdb/cropped_dataset/tezuka'
TEZUKA_DATASET_PATH='../Conditional-FastGAN/dataset/train_dataset/cropped_images/tezuka/'

MUCT_BIN_LMDB_PATH='dataset/lmdb/cropped_dataset/muct_bin/'
MUCT_BIN_DATASET_PATH='../Conditional-FastGAN/dataset/train_dataset/cropped_bi-thresh=40_can-thresh1=100_can-thresh2=170/muct/'


get_ipython().getoutput("mkdir -p $MUCT_LMDB_PATH $TEZUKA_LMDB_PATH")


get_ipython().getoutput("mkdir -p $MUCT_BIN_LMDB_PATH")


get_ipython().getoutput("python prepare_data.py --out $MUCT_LMDB_PATH --n_worker 2 --size 256 $MUCT_DATASET_PATH")


get_ipython().getoutput("python prepare_data.py --out $TEZUKA_LMDB_PATH --n_worker 2 --size 256 $TEZUKA_DATASET_PATH")


ls $TEZUKA_DATASET_PATH | wc -l


get_ipython().getoutput("python prepare_data.py --out $MUCT_BIN_LMDB_PATH --n_worker 2 --size 256 $MUCT_BIN_DATASET_PATH")



