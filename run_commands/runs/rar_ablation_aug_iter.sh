#SEED=1259051
MEM_SIZE=2000
NUM_TASKS=0
GPU_ID=0

NAME_PREFIX="run_commands/base_commands/command_"

for SEED in 1259051 1259052 1259053
do
  for DATASET_NAME in "mini_imagenet" "cifar100" "clrs25" "core50"
  do
  ALGO_NAME="er"

  RES_SIZE="reduced"


  for MEM_ITER in  1 2 5 10 20
  do
    ############### no augmentation
    FILE_NAME=$NAME_PREFIX$ALGO_NAME".sh"
    source $FILE_NAME $GPU_ID $NUM_TASKS $DATASET_NAME $SEED $MEM_SIZE $MEM_ITER $RES_SIZ

    ############ augmentation(P=1,Q=14) #########
    RAUG_N=1
    RAUG_M=14
    RAUG_TARGET="both"  ## mem incoming none
    MEM_BATCH=10
    FILE_NAME=$NAME_PREFIX$ALGO_NAME"_raug.sh"
    source $FILE_NAME $GPU_ID $NUM_TASKS $DATASET_NAME $SEED $MEM_SIZE $MEM_ITER \
    $RAUG_N $RAUG_M $RAUG_TARGET $MEM_BATCH $RES_SIZE

    ############ augmentation(P=2,Q=14) #########
    RAUG_N=2
    RAUG_M=14
    RAUG_TARGET="both"  ## mem incoming none
    MEM_BATCH=10
    FILE_NAME=$NAME_PREFIX$ALGO_NAME"_raug.sh"
    source $FILE_NAME $GPU_ID $NUM_TASKS $DATASET_NAME $SEED $MEM_SIZE $MEM_ITER \
    $RAUG_N $RAUG_M $RAUG_TARGET $MEM_BATCH $RES_SIZE

  done




done

done