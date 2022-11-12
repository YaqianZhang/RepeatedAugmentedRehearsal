

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
EPOCH=${12}
DER_alpha=${13}

 ## ER
               python general_main.py --data $DATASET_NAME --cl_type nc \
--agent "DER" --retrieve "random" --update random  \
--mem_size $MEM_SIZE  --eps_mem_batch $MEM_BATCH \
 --dataset_random_type task_random  --seed $SEED --num_tasks $NUM_TASKS \
 --seed $SEED --GPU_ID $GPU_ID --mem_iters $MEM_ITER \
--nmc_trick True \
--randaug False --randaug_N $RAUG_N  --randaug_M $RAUG_M --aug_target $RAUG_TARGET \
--deraug True \
--resnet_size $RES_SIZE --dataset_random_type "task_random" --epoch $EPOCH \
#--DER_alpha $DER_alpha
#--immediate_evaluate True
