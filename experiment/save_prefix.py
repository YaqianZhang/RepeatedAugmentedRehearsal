
import os
import time
def get_prefix_time(params,):

    folder_path = "results/" + str(params.seed)
    if (not os.path.exists(folder_path)):
        os.mkdir(folder_path)
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    trick = str(params.num_val)+"_"+timestamp
    prefix = folder_path + '/' + params.agent +str(params.epoch)+ "_" + params.retrieve[:3] + "_" + params.update[:3] + '_' + trick  + str(
        params.num_tasks) + "_" + str(params.mem_size)+ "_"+params.data+"_"


    print("save file name :" + prefix)

    return prefix
def RL_name(params,trick):
    if(params.aug_action_num != 8):
        trick += "aug"+str(params.aug_action_num)+"_"
    if(params.bpg_restart):
        trick += "restart_"
    if (params.RL_type not in ['NoRL','DormantRL'] and params.actor_type != "random"):
        trick+=params.hp_action_space+"_"
        if(params.hp_action_space in ["iter", "ratio_iter","aug_iter"]):
            trick += str(params.mem_iter_max) + str(params.mem_iter_min) + "_"

        if(params.virtual_update_times != 0):
            trick += "virtual"+str(params.virtual_update_times)+"_"
        if(params.use_ref_model):
            trick += "ref_"
        if(params.gamma != 1):
            trick += "g"+str(params.gamma)+"_"


        #reward
        trick += params.reward_type+"_"
        trick += str(params.reward_rg)+"_"
        if(params.reward_within_batch):
            trick += "wthin_"

        ## action
        if(params.action_space_type != "sparse"):
            trick += params.action_space_type+"_"

        ## state
        trick += params.state_feature_type+"_"

        ## dynamics
        if(params.episode_type == "multi-step"):
            trick += "Done"+str(params.done_freq)+"_"
            if(params.double_DQN == False):
                trick += "nodouble_"
        # if(params.dynamics_type == "next_batch"):
        #     trick+="nxtBch"+'_'
        if (params.dynamics_type == "within_batch"):
            trick += "wthBch" + '_'


        ## others

        # if (params.replay_old_only):
        #     trick += "oldonly" + "_"
        # if (params.split_new_old):
        #     trick += "splitno" + "_"
        # if(params.temperature_scaling):
        #     trick  += "TS_"

        if((params.RL_agent_update_flag==False)):
            trick +="NoT"+"_"

        if(params.RL_start_batchstep != -1 ):
            trick +="bstart"+str(params.RL_start_batchstep)+"_"
        # trick += str(params.task_start_mem_ratio)+str(params.task_start_incoming_ratio)+"_"



        trick += params.rl_exp_type+"_"
        ##critic_training
        if(params.RL_type == "RL_MDP_stop"):
            trick += params.immediate_reward[:2]+"_"
            if(params.stop_state_type != "4-dim"):
                trick += params.stop_state_type+"_"
            trick += "crt" + str(params.critic_layer_size) + "_" + str(params.critic_nlayer) + "_"
            # if(params.iter_penalty != -1):
            #     trick += str(params.iter_penalty)+"_"
            if (params.critic_use_model):
                trick += "Q"
                #if(params.preload_type != "cifar100"):
                trick += params.preload_type[:3]
                trick += "_"
            if(params.critic_ER_type != "recent2"):
                trick += params.critic_ER_type + "_"
        if(params.RL_type == "RL_MDP"):
            trick += "critic"+str(params.critic_layer_size)+"_"+str(params.critic_nlayer)+"_"
            # trick += "ERbch"+str(params.ER_batch_size)+"_"
            if(params.q_function_type != "lstm"):
                trick += "q"+params.q_function_type[:2]+"_"
            if(params.critic_type == "task_critic"):
                trick +="qtype"+params.critic_type[:1]+"_"

                trick+="t"+str(params.critic_task_layer)+"*"+str(params.critic_task_size)
                trick += "l" + str(params.critic_last_layer) + "*" + str(params.critic_last_size)+"_"
            if (params.critic_type == "actor_critic"):
                if(params.std_trainable):
                    trick += "std_"
                trick += "qtype" + "actor_"
                if(params.ratio_sigma != -1.01):
                    trick += "var"+str(params.ratio_sigma)+"_"
                if(params.actor_output_activation != "sigmoid"):
                    trick += "nosig_"
            if(params.ER_batch_size != 19):
                trick +="erb"+str(params.ER_batch_size)+"_"
            if(params.update_q_target_freq != 999):
                trick+="targetq"+str(params.update_q_target_freq)
            if(params.critic_use_model):
                trick += "Qmodel"+"_"

            trick += params.critic_ER_type+"_"
            if(params.critic_training_start != 79):
                trick += "qstart"+str(params.critic_training_start)+"_"

            if(params.critic_lr_type != "basic"):
                trick +="rllr"+params.critic_lr_type+"_"
            if(params.critic_wd >-1):
                trick +="wd-7"+"_"
            if(params.critic_lr != 0e-3):
                trick += str(params.critic_lr)
            # trick += "crtBchSize"+str(params.ER_batch_size)+"_"
            if(params.critic_training_iters != 0):
                trick += "crtitr" + str(params.critic_training_iters) + "_"
            if(params.critic_recent_steps != 99):
                trick += "criticRct"+str(params.critic_recent_steps)+"_"
            if(params.reward_test_type != "None"):
                trick += params.reward_test_type + "_"
    return trick
def DER_name(params,trick):
    if(params.agent[:3] == "DER"):
        if(params.DER_alpha != 0.3):
            trick += "a"+str(params.DER_alpha)+"_"
    if(params.agent in ["DER_head","DER_head_t1","DERPP_head_t1","DERPP_head"]):
        if(params.phead_layer != 1 or params.phead_size != 1024):
            trick += "phead"+str(params.phead_layer)+"x"+str(params.phead_size)+"_"
    return trick

def SCR_name(params,trick):

    if(params.agent[:3]=="SCR"):
        if(params.self_sup_beta > 0 ):
            trick += "beta"+str(params.self_sup_beta)+"_"
        trick+= "temp"+str(params.temp)+"_"
        if (params.softmax_type != 'None'):
            trick += "softmax"+str(params.softmax_nsize) \
                     +str(params.softmax_nlayers)
            if(params.softmax_membatch != 100):
                trick += str(params.softmax_membatch)
            if(params.softmax_dropout):
                trick += "dp"
            trick+="_"
            if(params.softmaxhead_lr != 0.1):
                trick +="smlr"+str(params.softmaxhead_lr) +"_"
    return trick
def aug_name(params,trick):
    if(params.adjust_aug_flag):
        trick +="daug_"
    if(params.adjust_iter_flag):
        trick +="diter_"


    if (params.aug_start > 0):
        trick += "q"+str(params.aug_start) + "_"

    # if (params.randaug_type == "dynamic"):
    #     trick += "raug_dyna_"
    # else:
    if ( params.agent in [ "ER_compress","ER_compress_both"]):
        trick += str(params.quality) + "_"
    if (params.randaug):
        trick += "raug"
        trick +=  str(params.randaug_N) + str(params.randaug_M) + "_"
        if (params.aug_target != "both"):
            trick += params.aug_target + "_"
    if(params.aug_normal):
        trick += "nml_"
    if(params.deraug):
        trick += "deraug_"
    if (params.scraug):
        trick += "scraug_"
        if (params.aug_target != "both"):
            trick += params.aug_target + "_"
    return trick


def get_prefix(params,run):

    trick = ""
    # if(params.batch != 10):
    #     trick += "b"+str(params.batch)+"_"
    if(params.resnet_size == "normal"):
        trick+="res_"
    if(params.immediate_evaluate):
        trick += "TEST_"
    if(params.ns_type != "noise"):
        trick += params.ns_type[:4]+"_"
    # if(params.joint_replay_type != "together"):
    #     trick += "replaySep_"

    # if(params.only_task_seen):
    #     trick+="onlySeen_"
    # if(params.frozen_old_fc):
    #     trick+="frz_"
    trick = aug_name(params,trick)
    if(params.batch != 10):
        trick += "B"+str(params.batch)+"_"
    if(params.bpg_lr != 5.0):
        trick += "blr"+str(params.bpg_lr) + "_"

    # if(params.online_hyper_tune):
    #     trick += "hp"+str(params.online_hyper_freq)+params.online_hyper_lr_list_type+"_"
    #     if(params.online_hyper_valid_type == "real_data"):
    #         trick += "real_"
    # else:
    #     if (params.learning_rate != 0.1):
    #         trick += "lr" + str(params.learning_rate) + "_"
    # if(params.agent == "SCR" and params.scr_memIter):
    #     trick += "memIter_"
    #     if(params.scr_memIter_type =="MAB"):
    #         trick += "MAB_"
    #     trick += params.scr_memIter_state_type +"_"
    #     trick += "act"+params.scr_memIter_action_type + "_"
    # if(params.lambda_ != 100):
    #     trick += str(params.lambda_)
    if(params.slide_window_size != 10):
        trick += "sw"+str(params.slide_window_size)+"_"
    if (params.nmc_trick):
        trick += "NMC_"
    # if (params.use_test_buffer):
    #     trick += "tbuf_"
    #     if(params.test_retrieve_num !=300):
    #         trick += "re"+str(params.test_retrieve_num)+"_"
    #     if(params.test_buffer_type == "reservior_sampling"):
    #         trick += "rs_"
    #     if(params.test_mem_type == "before"):
    #         trick += "bf" +"_"
    #     if(params.close_loop_mem_type == "low_acc"):
    #         trick+= "memlowacc_"
    # if (params.use_tmp_buffer):
    #     trick += "tmpMem_"

    ### scr relateed: temp, softmax ####
    trick = SCR_name(params,trick)
    trick = DER_name(params,trick)

    if(params.drift_detection):
        trick += "drf_"
    ### data augumentation ###
    # if (params.do_cutmix):
    #     #trick += "cmix"+str(params.cutmix_prob)+"_"+str(params.cutmix_batch)+"_"
    #     trick += "cmix"+"_"
    #     if(params.cutmix_type != "random"):
    #         trick +=params.cutmix_type +"_"


    if (params.no_aug):
        trick += "noaug_"
    # if(params.aug_type != ""):
    #     trick += params.aug_type
    # if (params.single_aug):
    #     trick += "saug_"


    ### adaptive CL
    if(params.adaptive_ratio_type == "online"):
        trick += "rs_"
    if(params.agent[:12] == "ER_dyna_iter"):
        trick += params.dyna_type[:3]+"_"+str(params.train_acc_max)+"_"+str(params.train_acc_min)
        trick += "_"+str(params.mem_iter_max)+"_"+str(params.mem_iter_min)+"_"
        trick += str(params.train_acc_max_aug) + "_" + str(params.train_acc_min_aug)+"_"
    if(params.dyna_mem_iter != "None"):

        trick += params.dyna_mem_iter+"_"

    if (params.dyna_mem_iter[:4] == "STOP"):
        trick += "stop" + str(params.stop_ratio) + "_"
    # if(params.agent == "ER_aug"):
    #     trick += str(params.train_acc_max_aug)


    if(params.iter_penalty != 0):
        trick += str(params.iter_penalty)+"_"

    if (params.mem_iters > 1):
        trick += "mIter" + str(params.mem_iters)+"_"
        if (params.start_mem_iters > -1):
            trick += "s"+str(params.start_mem_iters)+"_"




    if(params.agent[:8] == "ER_ratio"):
        pass
    else:
        if (params.incoming_ratio != 1):
            trick += "iratio" + str(params.incoming_ratio)+"_"
    if (params.mem_ratio != 1):
        trick += "mratio" + str(params.mem_ratio)+"_"
    # if(params.dyna_ratio != "None"):
    #     trick +="dyRatio"+params.dyna_ratio+"_"
    #
    #
    ## switch buffer
    if(params.switch_buffer_type != "one_buffer"):
        if(params.switch_buffer_type == "two_buffer"):
            trick += "2Buff"+"_"
        elif(params.switch_buffer_type == "dyna_buffer"):
            trick += "dBuff"+str(params.switch_buffer_freq)+"_"

        else:
            raise NotImplementedError("undefined switch buffer")
    #
    # if (params.test_mem_size != 300):
    #     trick += "tm" + str(params.test_mem_size) + "_"
    # if (params.test_mem_batchSize > 10):
    #     trick += "testBch" + str(params.test_mem_batchSize) + "_"



    # ## Rl related
    # if(params.mem_iter_max != 1):
    #
    #     trick += str(params.mem_iter_max)
    if (params.actor_type == "random"):
        trick += "rndRL_"
    trick = RL_name(params,trick)

    # if(params.test == "not_reset"):
    #     trick += "no_reset"


    if (not params.save_prefix == ""):
        trick += params.save_prefix+"_"

    if( not params.eps_mem_batch == 10):
        trick += "memBch"+str(params.eps_mem_batch)+"_"
    if(params.num_runs>1):
        trick += "numRuns"+str(params.num_runs) + "_"
    if(params.num_cycle >1):
        trick += "c"+str(params.num_cycle)+"_"


    if(params.dataset_random_type == "order_random"):
        trick += "orderRnd"+"_"
    trick += params.cl_type+"_"


    if(params.new_folder != ""):
        folder_path = "results/" + params.new_folder+"/"+str(params.seed)
        if (not os.path.exists("results/" + params.new_folder)):
            os.mkdir("results/" + params.new_folder)
    else:
        folder_path = "results/" + str(params.seed)
    if (not os.path.exists("results")):
        os.mkdir("results")
    if (not os.path.exists(folder_path)):
        os.mkdir(folder_path)
    prefix = folder_path + '/' + params.agent +str(params.epoch)+ "_" + params.retrieve[:3] + "_" + params.update[:3] + '_' + trick  + str(
        params.num_tasks) + "_" + str(params.mem_size)+ "_"+params.data+"_"
    print("save file name :"+ prefix)

    return prefix