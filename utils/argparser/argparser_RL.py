import argparse
from utils.utils import boolean_string


def parse_RL_para(parser):



    #################################### RL basics ####################################
    # parser.add_argument("--RL_type",dest='RL_type',default="NoRL",type=str,choices=[ "RL_actor","RL_ratio_1para","RL_adpRatio","RL_ratio",
    #                                                             "RL_memIter","NoRL","DormantRL","RL_ratioMemIter","RL_2ratioMemIter"],#"1dim","2dim",
    #                     help='RL_memIter dynamic adjust memIteration; 1dim and 2dim employ MAB to adjust coef of retrieve index')
    #

    parser.add_argument("--RL_type", dest='RL_type', default="NoRL", type=str,
                        choices=["RL_MDP_stop", "RL_MDP", "RL_MAB", "NoRL", "DormantRL", ],  # "1dim","2dim",
                        help='')
    ## action
    parser.add_argument('--action_size', dest='action_size', default=11,
                        type=int,
                        help='Action size (default: %(default)s)')
    parser.add_argument('--actor_type', dest='actor_type', default="greedy",
                        type=str, )
    parser.add_argument("--std_trainable", default=False)
    parser.add_argument("--action_space_type", dest="action_space_type", default="sparse", type=str,
                        choices=["cont", "monly_dense", "ionly_dense", "ionly", "upper", "posneu", "sparse", "medium",
                                 "dense"])
    parser.add_argument("--hp_action_space", default="ratio_iter", choices=["ratio", "ratio_iter", "iter", "aug_iter"])
    parser.add_argument("--MAB_reward_len", default="100", type=int)
    ## reward
    parser.add_argument("--reward_type", dest='reward_type', default="test_acc", type=str,
                        choices=["test_loss_v_rlt", "test_alpha01_loss_acc", "test_loss_old", "test_loss_median",
                                 "test_loss_acc", "acc_diff", "test_loss_rlt", "test_loss", "scaled", "real_reward",
                                 "incoming_acc", "mem_acc", "test_acc", "test_acc_rlt", "test_acc_rg", "relative",
                                 "multi-step", "multi-step-0", "multi-step-0-rlt", "multi-step-0-rlt-loss"],
                        help='')
    parser.add_argument('--reward_rg', dest='reward_rg', default=0, type=float, help="param to for rward regularization")

    parser.add_argument('--reward_within_batch', default=False, type=boolean_string)

    parser.add_argument("--reward_test_type", dest='reward_test_type', default="None", type=str,
                        choices=["reverse", "relative", "None"],
                        help='')

    ## state
    parser.add_argument("--state_feature_type", dest='state_feature_type', default="train_test4", type=str,
                        # choices=["new_old6_overall_train","new_old5_overall","new_old5_scale","new_old_old4","new_old_old4_noi","new_old_old","new_old5_4time","new_old5_task","new_old5_incoming","new_old6mn_org","new_old6mn_incoming","new_old3","new_old6mnt","new_old7","new_old6mn","new_old6m","new_old6","new_old11","new_old9","new_old5","new_old5t","new_old4","new_old2","3_dim", "4_dim", "3_loss", "4_loss", "6_dim",
                        #          "7_dim","task_dim","8_dim"],
                        help='state feature ')
    ## dynamics
    parser.add_argument("--done_freq", dest="done_freq", default=249, type=int)

    parser.add_argument("--virtual_update_times", default=0, type=int)
    parser.add_argument("--use_ref_model", default=False)

    parser.add_argument("--episode_type", dest='episode_type', default="batch", type=str, choices=["multi-step", "batch"],
                        help='')

    parser.add_argument("--dynamics_type", dest='dynamics_type', default="next_batch", type=str,
                        choices=["same_batch", "next_batch", "within_batch"],
                        help='whether the reward and transition dynamics are computed for same incoming batch or not')

    parser.add_argument("--RL_start_batchstep", dest="RL_start_batchstep", default=0, type=int)
    parser.add_argument("--RL_agent_update_flag", dest="RL_agent_update_flag", default=True, type=boolean_string)
    parser.add_argument("--start_task", default=0)
    parser.add_argument("--ratio_sigma", default=0.01)
    #################################### critic training####################################
    parser.add_argument('--q_function_type', type=str, default="mlp")
    parser.add_argument("--update_q_target_freq", default=250, type=int)
    parser.add_argument('--double_DQN', default=True, type=boolean_string)

    parser.add_argument("--critic_type", dest='critic_type', default='critic', type=str,
                        choices=["task_critic", "critic", "actor_critic", ])
    parser.add_argument("--actor_output_activation", default="sigmoid", choices=["sigmoid", "relu", "identity"])
    parser.add_argument("--critic_task_layer", default=0, type=int)
    parser.add_argument("--critic_last_layer", default=0, type=int)
    parser.add_argument("--critic_task_size", default=10, type=int)
    parser.add_argument("--critic_last_size", default=10, type=int)

    parser.add_argument("--critic_ER_type", dest='critic_ER_type', default='recent2', type=str,
                        choices=["recent4", "random", "recent", "recent2", "recent3"])

    parser.add_argument("--ER_batch_size", dest="ER_batch_size", default=50, type=int, )  # 50
    parser.add_argument('--critic_nlayer', dest='critic_nlayer', default=3,
                        type=int,
                        help='critic network size (default: %(default)s)')
    parser.add_argument('--critic_layer_size', dest='critic_layer_size', default=32,
                        type=int,
                        help='critic network size (default: %(default)s)')
    parser.add_argument('--critic_training_iters', dest='critic_training_iters', default=1,
                        type=int,
                        help="")

    parser.add_argument('--critic_lr', dest='critic_lr', default=5 * 10 ** (-4),
                        type=float,
                        help="")
    parser.add_argument("--critic_lr_type", dest="critic_lr_type", default="static", type=str,
                        choices=["static", "basic", "large", "mid", "small"])

    # parser.add_argument('--critic_wd', dest='critic_wd', default=0,
    #                     type=int,
    #                     help="")
    parser.add_argument('--critic_wd', dest='critic_wd', default=1 * 10 ** (-4),
                        type=int,
                        help="")
    parser.add_argument('--critic_training_start', dest='critic_training_start', default=80,
                        type=int,
                        help="")

    parser.add_argument('--critic_recent_steps', dest='critic_recent_steps', default=250,
                        type=int,
                        help="")
    parser.add_argument('--critic_use_model', dest='critic_use_model', default=False, type=boolean_string,
                        help="")
    parser.add_argument("--preload_type",default="cifar100",choices=["cifar100","mini_imagenet"])

    parser.add_argument("--rl_exp_type", dest="rl_exp_type", type=str, default="exp",
                        choices=["stb2", "l_exp", "stb", "exp", "m_exp3", "m_exp", "m_exp2"])

    ################### RL test buffer #####################
    parser.add_argument('--test_mem_size', dest='test_mem_size', default=300,
                        type=int,
                        help='Test Memory buffer size (default: %(default)s)')
    parser.add_argument('--test_mem_batchSize', dest='test_mem_batchSize', default=100,
                        type=int,
                        help='Test Memory buffer batch size (default: %(default)s)')
    parser.add_argument("--use_test_buffer",dest='use_test_buffer',default=False,type=boolean_string,
                        help='If True, evaluate model on the test buffer during CL training')
    parser.add_argument("--test_buffer_type",default="class_balance",choices=["class_balance","reservior_sampling"])
    parser.add_argument("--test_retrieve_num",default=300,type=int)
    parser.add_argument('--use_tmp_buffer',dest='use_tmp_buffer',default=False,type=boolean_string,
                        help='If True, use a tmp buffer to store the to-be-insert samples from new task/replace indices '
                             'and insert these into memory at the end of new task')

    parser.add_argument('--strict_balance', default="False", type=boolean_string,
                        help="whether computing state stats on a class balanced sample from train memory and test memory")
    parser.add_argument("--test_mem_type", dest='test_mem_type', default="after", type=str, choices=["before", "after"],
                        help='')
    parser.add_argument("--test_mem_recycle",default = False, type=boolean_string)


    #################### early stop with RL ##########
    parser.add_argument("--gamma",default=1,type=float)
    parser.add_argument("--immediate_reward",default="penalty",choices=["relative_train_acc_mem","mem_inc_ratio","train_acc","penalty","debug"])
    parser.add_argument("--iter_penalty",default=0,type=float)
    parser.add_argument("--stop_ratio",default=3,type=float)
    parser.add_argument("--stop_state_type",default="4-dim",choices=["4-dim","2-dim","5-dim"])
    return parser