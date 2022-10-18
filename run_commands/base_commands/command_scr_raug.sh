

GPU_ID=$1
NUM_TASKS=$2
DATASET_NAME=$3
SEED=$4
MEM_SIZE=$5
MEM_ITER=$6
RAUG_N=$7
RAUG_M=$8
RAUG_TARGET=$9
MEM_BATCH=${10}
RES_SIZE=${11}
 ## scr raug
             python general_main.py --data $DATASET_NAME --cl_type nc \
--agent "SCR" --retrieve random --update random  \
--mem_size $MEM_SIZE --head mlp --temp 0.07 --eps_mem_batch 100 \
 --dataset_random_type task_random  --seed $SEED --num_tasks $NUM_TASKS \
 --seed $SEED --GPU_ID $GPU_ID --mem_iters $MEM_ITER \
--nmc_trick True \
--randaug True --randaug_N $RAUG_N  --randaug_M $RAUG_M --aug_target $RAUG_TARGET \
--resnet_size $RES_SIZE
