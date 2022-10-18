#SEED=1259051
MEM_SIZE=2000

NUM_TASKS=0
GPU_ID=1
NAME_PREFIX="run_commands/base_command_"


for SEED in 1259051 1259052 1259053
do
  ALGO_NAME="er" ## er, scr, aser

for DATASET_NAME in      "cifar100" "mini_imagenet" "clrs25" "core50"
  do



  MEM_ITER=10
  RAUG_N=1
  RAUG_M=14
  RAUG_TARGET="both"  ## mem incoming none
  MEM_BATCH=100
  RES_SIZE="reduced"
  MEM_MAX=15
  STOP_RATIO=10
  LR_large=10
  LR_small=5
  ACC_MAX=0.9
  ACC_MIN=0.8
  AUG_NUM=5
  AUG_MAX=0.8
  AUG_MIN=0.7

  FILE_NAME=$NAME_PREFIX"_adaptive_rar_RL.sh"
  source $FILE_NAME $GPU_ID $NUM_TASKS $DATASET_NAME $SEED $MEM_SIZE $MEM_ITER \
  $RAUG_N $RAUG_M $RAUG_TARGET $MEM_BATCH $RES_SIZE $MEM_MAX $STOP_RATIO \
  $LR_large $LR_small $ACC_MAX $ACC_MIN $AUG_NUM $AUG_MAX $AUG_MIN
  #
  #

done

done