
from utils.utils import boolean_string

def parse_replay(parser):

    ################## Adaptive CL ###########################

    parser.add_argument("--adjust_iter_flag",default=False,type=boolean_string)
    parser.add_argument("--dyna_mem_iter",dest='dyna_mem_iter',default="None",type=str,choices=["random","STOP_loss","None","STOP_acc_loss","STOP_acc"],
                        help='If True, adjust mem iter')
    parser.add_argument("--train_acc_max",default=0.95,type=float)
    parser.add_argument("--train_acc_max_aug",default=0.90,type=float)
    parser.add_argument("--train_acc_min_aug",default=0.80,type=float)
    parser.add_argument("--train_acc_min",default=0.85,type=float)


    parser.add_argument('--mem_iter_max', dest='mem_iter_max', default=20, type=int,
                        help='')

    parser.add_argument('--mem_iter_min', dest='mem_iter_min', default=1, type=int,
                        help='')

    parser.add_argument("--dyna_type",default="train_acc",choices=["bpg","random","train_acc"])



    # #################### replay dynamics ####################
    parser.add_argument("--joint_replay_type",default="together",choices=["together","seperate"],
                        help="implementation type of joint training of incoming batch and memory batch")
    parser.add_argument("--online_hyper_tune", default=False, type=boolean_string)
    parser.add_argument("--online_hyper_valid_type", default="test_data", type=str, choices=["real_data","test_mem"])
    parser.add_argument("--online_hyper_freq", default=1, type=int)
    parser.add_argument("--online_hyper_lr_list_type",default="basic",choices=["scr","basic","4lr","5lr"])
    parser.add_argument("--online_hyper_RL",default=False,type=boolean_string)
    parser.add_argument("--scr_memIter", default=False, type=boolean_string)
    parser.add_argument("--scr_memIter_type",default="c_MAB",choices=["c_MAB","MAB"])
    parser.add_argument("--scr_memIter_state_type", default="4dim", choices=["7dim","6dim","3dim","4dim","train"])
    parser.add_argument("--scr_memIter_action_type", default="4", choices=["4","8"])

    # parser.add_argument("--temperature_scaling",default=False,type=boolean_string)
    # parser.add_argument("--frozen_old_fc", dest="frozen_old_fc", default=False, type=boolean_string)
    parser.add_argument("--close_loop_mem_type", default="random",
                        choices=["low_acc",  "random", ])

    parser.add_argument('--mem_ratio_max', default=1.5,
                        help='')

    parser.add_argument('--mem_ratio_min', default=0.1,
                        help='')
    parser.add_argument('--incoming_ratio', dest='incoming_ratio', default=1.0, type=float,
                        help='incoming  gradient update ratio')
    parser.add_argument('--mem_ratio', dest='mem_ratio', default=1.0, type=float,
                        help='mem gradient update ratio')

    parser.add_argument('--task_start_mem_ratio', dest='task_start_mem_ratio', default=0.5, type=float,
                        help='mem gradient update ratio')
    parser.add_argument('--task_start_incoming_ratio', dest='task_start_incoming_ratio', default=0.1, type=float,
                        help='mem gradient update ratio')
    parser.add_argument("--dyna_ratio", dest='dyna_ratio', type=str, default="None", choices=['dyna','random','None'],
                        help='adjust dyna_ratio')

    parser.add_argument("--adaptive_ratio_type",type=str,default="offline",choices=["online","offline",])
    return parser