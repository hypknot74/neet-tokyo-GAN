CUDA = 0
NAME = '0'
PATH = '../../dataset/processed_256'

# パラメータ
IM_SIZE=256
NZ = 256
NUM_INNER_ITERATIONS = 1
ITER = 100000


get_ipython().getoutput("CUDA_VISIBLE_DEVICES=$CUDA python ../train.py\")
--path $PATH --im_size $IM_SIZE  --batch_size 8 --name $NAME \
--num_inner_iterations $NUM_INNER_ITERATIONS --nz $NZ \
--iter $ITER \
--wandb


cd ./train_results/$NAME/


get_ipython().getoutput("MKL_SERVICE_FORCE_INTEL=CUDA python ../../../generate_video.py \")
--ckpt ./models/100000.pth --name $NAME --im_size $IM_SIZE --nz $NZ 



