

GPU_ID=$1
NUM_TASKS=$2
DATASET_NAME=$3
SEED=$4
MEM_SIZE=$5
MEM_ITER=$6
RES_SIZE=$7

 ## scr

             python general_main.py --data $DATASET_NAME --cl_type nc \
--agent "SCR" --retrieve random --update random  \
--mem_size $MEM_SIZE --head mlp --temp 0.07 --eps_mem_batch 100 \
 --dataset_random_type task_random  --seed $SEED --num_tasks $NUM_TASKS \
 --seed $SEED --GPU_ID $GPU_ID --mem_iters $MEM_ITER \
--nmc_trick True \
--resnet_size $RES_SIZE
