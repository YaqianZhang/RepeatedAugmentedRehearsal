

GPU_ID=$1
NUM_TASKS=$2
DATASET_NAME=$3
SEED=$4
MEM_SIZE=$5
MEM_ITER=$6
RES_SIZE=$7

 ## ASER

               python general_main.py --data $DATASET_NAME --cl_type nc \
--agent "ER" --retrieve "ASER" --update "ASER"  \
--mem_size $MEM_SIZE  --eps_mem_batch 10 \
 --dataset_random_type task_random  --seed $SEED --num_tasks $NUM_TASKS \
 --seed $SEED --GPU_ID $GPU_ID --mem_iters $MEM_ITER  \
 --aser_type "asvm" --n_smp_cls 1.5 --k 3 --nmc_trick True \
--resnet_size $RES_SIZE