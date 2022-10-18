

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
MEM_MAX=${12}
STOP_RATIO=${13}
BPG_LR_large=${14}
BPG_LR_small=${15}
ACC_MAX=${16}
ACC_MIN=${17}
AUG_NUM=${18}
AUG_MAX=${19}
AUG_MIN=${20}
 ## ER-aug-dyna

               python general_main.py --data $DATASET_NAME --cl_type nc \
--agent "ER_dyna_iter_aug_dbpg_joint" --retrieve "random" --update random  \
--mem_size $MEM_SIZE  --eps_mem_batch $MEM_BATCH \
 --dataset_random_type task_random  --seed $SEED --num_tasks $NUM_TASKS \
 --seed $SEED --GPU_ID $GPU_ID --mem_iters $MEM_ITER \
--nmc_trick True \
--randaug True --randaug_N $RAUG_N  --randaug_M $RAUG_M --aug_target $RAUG_TARGET \
--resnet_size $RES_SIZE  --dyna_type "bpg"  --mem_iter_max $MEM_MAX \
--dyna_mem_iter "STOP_loss" --stop_ratio $STOP_RATIO --bpg_restart True \
--adjust_aug_flag True --bpg_lr_large $BPG_LR_large --bpg_lr_small $BPG_LR_small \
--train_acc_max $ACC_MAX --train_acc_min $ACC_MIN --save_prefix "nostopflag" \
--aug_action_num $AUG_NUM --train_acc_max_aug $AUG_MAX --train_acc_min_aug $AUG_MIN
