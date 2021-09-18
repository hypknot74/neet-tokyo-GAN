PATH='../../dataset/lmdb_1024'
CKPT='../official_checkpoint/stylegan2-ffhq-config-f.pt'
ITER=50000
SIZE=1024


get_ipython().getoutput("python ../train.py --ckpt $CKPT --iter $ITER --size $SIZE $PATH")



