#SEED=1259051
MEM_SIZE=2000
NUM_TASKS=0
#DATASET_NAME="cifar10"
GPU_ID=0
NAME_PREFIX="run_commands/base_commands/command_"

RES_SIZE="reduced"
for DATASET_NAME in "cifar100"
do
for SEED in 1259051 1259052   1259053
do
  for ALGO_NAME in  "er"
  do
   ########### baseline: without RAR
  MEM_ITER=1
  FILE_NAME=$NAME_PREFIX$ALGO_NAME".sh"
  source $FILE_NAME $GPU_ID $NUM_TASKS $DATASET_NAME $SEED $MEM_SIZE $MEM_ITER $RES_SIZE


#
   ########### with RAR
  MEM_ITER=1
  RAUG_N=1
  RAUG_M=14
  RAUG_TARGET="both"  ## mem incoming none
  MEM_BATCH=10
  FILE_NAME=$NAME_PREFIX$ALGO_NAME"_raug.sh"
  source $FILE_NAME $GPU_ID $NUM_TASKS $DATASET_NAME $SEED $MEM_SIZE $MEM_ITER \
  $RAUG_N $RAUG_M $RAUG_TARGET $MEM_BATCH $RES_SIZE
#


  done
  #
  #


done

done