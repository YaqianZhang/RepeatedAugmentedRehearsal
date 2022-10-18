import argparse
import random
import numpy as np
import torch
from experiment.run import multiple_run,multiple_RLtrainig_run
from utils.utils import boolean_string
import warnings
from utils.argparser.argparser_RL import parse_RL_para
from utils.argparser.argparser_basic import parse_cl_basic
from utils.argparser.argparser_scr import parse_scr
from utils.argparser.argparser_aug import parse_aug
from utils.argparser.argparser_replay import parse_replay


warnings.filterwarnings("ignore", category=DeprecationWarning)



def main(args):
    print(args)
    # set up seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    args.trick = {'labels_trick': args.labels_trick, 'separated_softmax': args.separated_softmax,
                  'kd_trick': args.kd_trick, 'kd_trick_star': args.kd_trick_star, 'review_trick': args.review_trick,
                  'nmc_trick': args.nmc_trick}
    if(args.num_runs>1):
        multiple_RLtrainig_run(args)
    else:
        multiple_run(args)



if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Online Continual Learning PyTorch")

    parser = parse_cl_basic(parser)
    parser = parse_scr(parser)

    parser = parse_RL_para(parser)
    parser = parse_aug(parser)
    parser = parse_replay(parser)


    parser.add_argument('--aug_action_num',default=8,type=int)
    ######################### misc ########################
    parser.add_argument("--immediate_evaluate",default = False, type = boolean_string)
    parser.add_argument('--dataset_random_type', dest='dataset_random_type', default= "order_random",
                        type=str,choices=["order_random","task_random"],
                        help="")
    parser.add_argument("--resnet_size",default="reduced",choices=["normal","reduced"])
    parser.add_argument('--save_prefix', dest='save_prefix', default="",  help='')

    parser.add_argument('--new_folder', dest='new_folder', default="", help='')

    parser.add_argument('--bpg_restart',default=False,type=boolean_string)
    parser.add_argument('--test', dest='test', default=" ", type=str,choices=["not_reset"],
                        help='')
    parser.add_argument('--debug_mode',default=False, type=boolean_string)
    parser.add_argument('--acc_no_aug',default=True)

    parser.add_argument('--GPU_ID', dest='GPU_ID', default= 0,
                        type=int,
                        help="")
    parser.add_argument('--drift_detection',type=boolean_string,default=False)
    parser.add_argument("--test_add_buffer",default=False,type=boolean_string)

    parser.add_argument('--switch_buffer_type', dest='switch_buffer_type', default="one_buffer",
                        type=str, choices=["one_buffer", "two_buffer", "dyna_buffer"],
                        help="whether and how to switch replay buffer")
    #parser.add_argument("--adjust_aug_flag",default=False,type=boolean_string)
    ############ thompson sampling ###########
    parser.add_argument("--slide_window_size",default=10,type=int)

    parser.add_argument("--set_task_flag",default=-1,type=int)
    parser.add_argument("--bpg_lr",default=5.0,type=float)
    parser.add_argument("--bpg_lr_large",default=10.0,type=float)
    parser.add_argument("--bpg_lr_small",default=5.0,type=float)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    torch.cuda.set_device(args.GPU_ID)#args.GPU_ID

    if(args.data=="cifar100"):
        if(args.set_task_flag>-1):
            args.num_tasks = args.set_task_flag
        else:
            args.num_tasks = 20
    elif(args.data=="cifar10"):
        args.num_tasks = 5
    elif(args.data=="mini_imagenet"):
        args.num_tasks=10
    elif(args.data=="clrs25"):
        if(args.cl_type == "nc"):
            args.num_tasks=5
        else:
            args.num_tasks=3
    elif(args.data=="core50"):
        args.num_tasks=9
    else:
        raise NotImplementedError("not seen dataset",args.data)

    main(args)
# ######### meta learn lamaml related
#     parser.add_argument('--learn_lr',type=boolean_string,default = False,help="whether to learn softmax head lr")
#     parser.add_argument('--second_order',type=boolean_string,default= False)
#     parser.add_argument('--grad_clip_norm', type=float, default=2.0,
#                         help='Clip the gradients by this value')
#     parser.add_argument('--sync_update', default=False, action='store_true',
#                         help='the LRs and weights should be updated synchronously')
#     parser.add_argument('--xav_init', default=False, action='store_true',
#                         help='Use xavier initialization')
#     parser.add_argument('--opt_lr', type=float, default=1e-1,
#                         help='learning rate for LRs')
#     parser.add_argument('--opt_wt', type=float, default=1e-1,
#                         help='learning rate for weights')
#     parser.add_argument('--alpha_init', type=float, default=1e-3,
#                         help='initialization for the LRs')


# ################### LAMAML
# parser.add_argument('--alpha_init', type=float, default=1e-3,
#                     help='initialization for the LRs')
# #################################### multiple buffer idea ####################################
#
# parser.add_argument('--test_retrieval_step', dest='test_retrieval_step', default= -1,
#                     type=int,
#                     help="")
#
#
#
# parser.add_argument('--switch_buffer_freq', dest='switch_buffer_freq', default= 1000,
#                     type=int,
#                     help="")
#
# parser.add_argument("--only_task_seen",dest="only_task_seen",default=False,type=boolean_string)

# parser.add_argument("--replay_old_only",dest="replay_old_only",default=False,type=boolean_string,)
#
# parser.add_argument("--split_new_old",dest="split_new_old",default=False,type=boolean_string)

# parser.add_argument('--mem_iter_std', dest='mem_iter_std', default=0.3, type=int,
#                     help='')
#
#
