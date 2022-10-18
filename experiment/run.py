import time
import numpy as np
from continuum.continuum import continuum
from continuum.data_utils import setup_test_loader
from utils.name_match import agents
from utils.setup_elements import setup_opt, setup_architecture
from utils.utils import maybe_cuda
from experiment.metrics import compute_performance, single_run_avg_end_fgt
from experiment.tune_hyperparam import tune_hyper
from types import SimpleNamespace
from utils.io import load_yaml, save_dataframe_csv, check_ram_usage
import pandas as pd
import os
import pickle
#from RL.evaluator import evaluator
import torch
from experiment.save_prefix import get_prefix





def save_stats_acc(params,accuracy_list,run=1,loss_list=[]):
    prefix = "results/"+str(params.seed) + "/" +params.exp_name+"_"+params.agent+"_"+params.data+"_"+ "tune"

    #print("acc_zyq",accuracy_list) #+str(params.eps_mem_batch)+
    np.save(prefix + "accuracy_list.npy", accuracy_list)
    np.save(prefix + "loss_list.npy", loss_list)



def save_stats(params,agent,model,accuracy_list,running_time,run=1,immediate_acc_list=[]):

    prefix = get_prefix(params,run)

    np.save(prefix + "running_time.npy", running_time)


    #print("acc_zyq",accuracy_list) #+str(params.eps_mem_batch)+
    np.save(prefix + "accuracy_list.npy", accuracy_list)
    #np.save(prefix + "loss_list.npy", loss_list)
    np.save(prefix + "immediate_acc_list.npy", immediate_acc_list)


    # if(params.batch != 9):
    #     trick += str(params.batch)+"_"
    if (params.use_test_buffer):
        if(params.agent == "RLER"):
            pass
        else:
            agent.close_loop_cl.save_task_reward(prefix)
            print("save task!!!!!!!!!!!!!!")
    agent.save_training_acc(prefix)  # training_accuracy

    if(params.agent== 'ER' or params.agent == "ICARL"):
        agent.buffer.save_buffer_info(prefix)
   # # if( params.RL_type != "NoRL" ):
   #  if(agent.RL_replay != None and agent.RL_replay != "NoRL"):
   #      print("save reward in run")
   #      agent.RL_replay.RL_agent.save_RL_stats(prefix)  # q, reward, action
   #      agent.RL_replay.RL_agent.ExperienceReplayObj.save_buff(prefix)
        # if (params.online_hyper_RL or params.scr_memIter or params.agent in ["SCR_RL_ratio","SCR_RL_iter",
        #                                                                      "ER_RL_ratio","ER_RL_iter",
        #                                                                      "ER_RL_addIter","ER_dyna_iter"]):
        #     agent.RL_replay.RL_agent.save_RL_stats(prefix)  # q, reward, action
        #
        # else:
        #     agent.RL_agent.save_RL_stats(prefix) # q, reward, action

            #agent.RL_env.save_task_reward(prefix)

    agent.save_mem_iters(prefix) ## memiter raio

def reset_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def save_task_info(params,data_continuum,run):

    prefix = get_prefix(params,run)
    # task_label = np.array(data_continuum.data_object.task_labels)
    # np.save(prefix + "task_label.npy", task_label)
    save_obj(params,prefix+"para")




def multiple_run(params,store=False,save_path=False):
    # Set up data stream
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(params.data, params.cl_type, params)

    # params['train_steps_total'] = int(2500/ 10) * params.num_tasks *params.num_runs


    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))

    if store:
        result_path = load_yaml('config/global.yml', key='path')['result']
        table_path = result_path + params.data
        print(table_path)
        os.makedirs(table_path, exist_ok=True)
        if not save_path:
            save_path = params.model_name + '_' + params.data_name + '.pkl'

    accuracy_list = []
    for run in range(params.num_runs):
        tmp_acc = []
        tmp_acc_immediate = []
        run_start = time.time()
        data_continuum.new_run()

        model = setup_architecture(params)
        model = maybe_cuda(model, params.cuda)
        opt = setup_opt(params.optimizer, model, params.learning_rate, params.weight_decay)
        agent = agents[params.agent](model, opt, params)




        # prepare val data loader
        test_loaders = setup_test_loader(data_continuum.test_data(), params)
        agent.test_loaders = test_loaders


        # for task, test_loader in enumerate(test_loaders):
        #     for i, (batch_x, batch_y) in enumerate(test_loader):
        #         print(task,i,torch.unique(batch_y))

        save_task_info(params,data_continuum,run)

        # if(params.reward_type == "real_reward" or params.online_hyper_valid_type == "real_data"):
        #     agent.evaluator = evaluator(test_loaders)
        # else:
        #     agent.evaluator = None

        for cycle in range(params.num_cycle):
            data_continuum.cur_task=0

            for i, (x_train, y_train, labels) in enumerate(data_continuum):
                # if( i>1):
                #     break  ## debug


                print("-----------run {} cycle{} training task {}-------------".format(run,cycle, i))
                print('task '+str(i)+' size: {}, {}'.format(x_train.shape, y_train.shape))


                immediate_acc_list = agent.train_learner(x_train, y_train,)
               # agent.train_learner(x_train, y_train, labels)
                acc_array = agent.evaluate(test_loaders)
                tmp_acc.append(acc_array)
                tmp_acc_immediate.append(immediate_acc_list)

                # if (params.RL_type != "NoRL"):
                # if (params.use_test_buffer):
                #
                #     if (params.agent == "RLER"):
                #         pass
                #     else:
                #         print("update",agent.close_loop_cl.task_tacc)
                #
                #         agent.close_loop_cl.update_task_reward()


            run_end = time.time()
            print(
                "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, np.mean(tmp_acc[-1]),
                                                                               run_end - run_start))
            accuracy_list.append(np.array(tmp_acc))
    accuracy_list = np.array(accuracy_list)  # 1*task num*2*task_num
    print(accuracy_list.shape)
    end = time.time()
    running_time = end-start

    save_stats(params, agent, model,accuracy_list,running_time,immediate_acc_list)

    avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_list[:,:,0,:])
    end = time.time()

    print('----------- Total {} run: {}s -----------'.format(params.num_runs, end - start))
    print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
          .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))

def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)


def multiple_RLtrainig_run(params):
    # Set up data stream
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(params.data, params.cl_type, params)


    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))
    accuracy_list = []
    loss_list=[]
    model = setup_architecture(params)
    model = maybe_cuda(model, params.cuda)
    opt = setup_opt(params.optimizer, model, params.learning_rate, params.weight_decay)
    agent = agents[params.agent](model, opt, params)
    for run in range(params.num_runs):
        tmp_acc = []
        tmp_loss=[]
        run_start = time.time()
        data_continuum.new_run()
        # initailize agent model

        # agent.initialize_agent(params)
        agent.task_seen =0
        # if (params.RL_type != "NoRL"):
        #     agent.RL_env.initialize()
        #print("buffer index",agent.buffer.current_index,agent.RL_env.test_buffer.current_index)

        model = setup_architecture(params)
        model = maybe_cuda(model, params.cuda)
        opt = setup_opt(params.optimizer, model, params.learning_rate, params.weight_decay)
        agent = agents[params.agent](model, opt, params)


        # prepare val data loader
        test_loaders = setup_test_loader(data_continuum.test_data(), params)
        save_task_info(params,data_continuum,0)

        # if (params.reward_type == "real_reward" or params.online_hyper_valid_type == "real_data"):
        #
        #     agent.evaluator = evaluator(test_loaders)
        # else:
        #     agent.evaluator = None

        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if(params.debug_mode and i>2):
                break  ## debug


            print("-----------run {} training task {}-------------".format(run, i))
            print('task '+str(i)+' size: {}, {}'.format(x_train.shape, y_train.shape))

            agent.train_learner(x_train, y_train,)
            acc_array = agent.evaluate(test_loaders)
            tmp_acc.append(acc_array)
            #mp_loss.append(loss_array)
            # if (params.RL_type != "NoRL"):
            # if(params.use_test_buffer):
            #
            #     agent.close_loop_cl.update_task_reward()
        run_end = time.time()
        print(
            "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, np.mean(tmp_acc[-1]),
                                                                           run_end - run_start))
        accuracy_list.append(np.array(tmp_acc))
        #loss_list.append(np.array(tmp_loss))
        #if(run%3==0):
        accuracy_list_arr = np.array(accuracy_list)
        #loss_list_arr = np.array(loss_list)
        save_stats(params, agent, model, accuracy_list_arr,run,)

    accuracy_list = np.array(accuracy_list)
    #loss_list = np.array(loss_list)
    end = time.time()
    running_time = end-start
    save_stats(params, agent, model,accuracy_list,running_time,run,loss_list)

    #avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_list)

    avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_list[:,:,0,:])

    accuracy_array = np.array(accuracy_list)

    end = time.time()




def multiple_run_tune(defaul_params, tune_params, save_path):
    # Set up data stream
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(defaul_params.data, defaul_params.cl_type, defaul_params)
    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))

    #store table
    # set up storing table
    table_path = load_yaml('config/global.yml', key='path')['tables']
    metric_list = ['Avg_End_Acc'] + ['Avg_End_Fgt'] + ['Time'] + ["Batch" + str(i) for i in range(defaul_params.num_val, data_continuum.task_nums)]
    param_list = list(tune_params.keys()) + metric_list
    table_columns = ['Run'] + param_list
    table_path = table_path + defaul_params.data
    os.makedirs(table_path, exist_ok=True)
    if not save_path:
        save_path = defaul_params.model_name + '_' + defaul_params.data_name + '.csv'
    df = pd.DataFrame(columns=table_columns)
    # store list
    accuracy_list = []
    params_keep = []
    for run in range(defaul_params.num_runs):
        tmp_acc = []
        tune_data = []
        run_start = time.time()
        data_continuum.new_run()
        # prepare val data loader
        test_loaders = setup_test_loader(data_continuum.test_data(), defaul_params)
        tune_test_loaders = test_loaders[:defaul_params.num_val]
        test_loaders = test_loaders[defaul_params.num_val:]
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < defaul_params.num_val:
                #collection tune data
                tune_data.append((x_train, y_train, labels))
                if len(tune_data) == defaul_params.num_val:
                    # tune
                    best_params = tune_hyper(tune_data, tune_test_loaders, defaul_params, tune_params)
                    params_keep.append(best_params)
                    final_params = vars(defaul_params)
                    final_params.update(best_params)
                    final_params = SimpleNamespace(**final_params)
                    # set up
                    print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
                    model = setup_architecture(final_params)
                    model = maybe_cuda(model, final_params.cuda)
                    opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
                    agent = agents[final_params.agent](model, opt, final_params)
                    print('Training Start')
            else:
                print("----------run {} training batch {}-------------".format(run, i))
                print('size: {}, {}'.format(x_train.shape, y_train.shape))
                agent.train_learner(x_train, y_train)
                acc_array,loss_array = agent.evaluate(test_loaders)
                tmp_acc.append(acc_array)

        run_end = time.time()
        print(
            "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, np.mean(tmp_acc[-1]),
                                                                           run_end - run_start))
        accuracy_list.append(np.array(tmp_acc))
        save_stats_acc(defaul_params, agent, model, accuracy_list)

        #store result
        result_dict = {'Run': run}
        result_dict.update(best_params)
        end_task_acc = tmp_acc[-1]
        for i in range(data_continuum.task_nums - defaul_params.num_val):
            result_dict["Batch" + str(i + defaul_params.num_val)] = end_task_acc[i]
        result_dict['Avg_End_Acc'] = np.mean(tmp_acc[-1])
        result_dict['Avg_End_Fgt'] = single_run_avg_end_fgt(np.array(tmp_acc))
        result_dict['Time'] = run_end - run_start
        df = df.append(result_dict, ignore_index=True)
        save_dataframe_csv(df, table_path, save_path)

    accuracy_list = np.array(accuracy_list)
    avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_list)
    end = time.time()
    final_result = {'Run': 'Final Result'}
    final_result['Avg_End_Acc'] = avg_end_acc
    final_result['Avg_End_Fgt'] = avg_end_fgt
    final_result['Time'] = end - start
    df = df.append(final_result, ignore_index=True)
    save_dataframe_csv(df, table_path, save_path)
    print('----------- Total {} run: {}s -----------'.format(defaul_params.num_runs, end - start))
    print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
          .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))



def multiple_run_tune_separate(default_params, tune_params, save_path):
    # Set up data stream
    # print(default_params.GPU_ID)
    # assert False
    #args.cuda = torch.cuda.is_available()
    torch.cuda.set_device(default_params.GPU_ID)#args.GPU_ID
    print("~~~~~~~~ run tune seperate")
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(default_params.data, default_params.cl_type, default_params)
    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))

    if default_params.num_val == -1:
        # offline tuning
        default_params.num_val = data_continuum.data_object.task_nums
    #store table
    # set up storing table
    result_path = load_yaml('config/global.yml', key='path')['result']
    table_path = result_path + default_params.data + '/' + default_params.cl_type
    for i in default_params.trick:
        if default_params.trick[i]:
            trick_name = i
            table_path = result_path + default_params.data + '/' + default_params.cl_type + '/' + trick_name
            break
    print(table_path)
    os.makedirs(table_path, exist_ok=True)
    if not save_path:
        save_path = default_params.model_name + '_' + default_params.data_name + '_' + str(default_params.seed) + '.pkl'
    # store list
    accuracy_list = []
    params_keep = []
    if isinstance(default_params.num_runs, int):
        run_list = range(default_params.num_runs)
    else:
        run_list = default_params.num_runs
    for run in run_list:
        tmp_acc = []
        run_start = time.time()
        data_continuum.new_run()
        if default_params.train_val:
            single_tune_train_val(data_continuum, default_params, tune_params, params_keep, tmp_acc, run)
        else:
            single_tune(data_continuum, default_params, tune_params, params_keep, tmp_acc, run)
        run_end = time.time()
        print(
            "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, np.mean(tmp_acc[-1]),
                                                                           run_end - run_start))
        accuracy_list.append(np.array(tmp_acc))
        save_stats_acc(default_params, accuracy_list)

    end = time.time()
    accuracy_array = np.array(accuracy_list)
    result = {'seed': default_params.seed}
    result['time'] = end - start
    result['acc_array'] = accuracy_array
    result['ram'] = check_ram_usage()
    result['best_params'] = params_keep
    save_file = open(table_path + '/' + save_path, "wb")
    pickle.dump(result, save_file)
    save_file.close()
    print('----------- Total {} run: {}s -----------'.format(default_params.num_runs, end - start))
    print('----------- Seed {} RAM: {}s -----------'.format(default_params.seed, result['ram']))

def single_tune(data_continuum, default_params, tune_params, params_keep, tmp_acc, run):
    tune_data = []
    # prepare val data loader
    test_loaders_full = setup_test_loader(data_continuum.test_data(), default_params)
    tune_test_loaders = test_loaders_full[:default_params.num_val]
    test_loaders = test_loaders_full[default_params.num_val:]

    if default_params.online:
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                tune_data.append((x_train, y_train, labels))
                if len(tune_data) == default_params.num_val:
                    # tune
                    best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
                    params_keep.append(best_params)
                    final_params = vars(default_params)
                    final_params.update(best_params)
                    final_params = SimpleNamespace(**final_params)
                    # set up
                    print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
                    model = setup_architecture(final_params)
                    model = maybe_cuda(model, final_params.cuda)
                    opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
                    agent = agents[final_params.agent](model, opt, final_params)
                    print('Training Start')
            else:
                print("----------run {} training batch {}-------------".format(run, i))
                print('size: {}, {}'.format(x_train.shape, y_train.shape))
                agent.train_learner(x_train, y_train)
                acc_array = agent.evaluate(test_loaders)
                tmp_acc.append(acc_array)
    else:
        x_train_offline = []
        y_train_offline = []
        x_tune_offline = []
        y_tune_offline = []
        labels_offline = []
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                x_tune_offline.append(x_train)
                y_tune_offline.append(y_train)
                labels_offline.append(labels)
            else:
                x_train_offline.append(x_train)
                y_train_offline.append(y_train)
        tune_data = [(np.concatenate(x_tune_offline, axis=0), np.concatenate(y_tune_offline, axis=0),
                      np.concatenate(labels_offline, axis=0))]
        best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
        params_keep.append(best_params)
        final_params = vars(default_params)
        final_params.update(best_params)
        final_params = SimpleNamespace(**final_params)
        # set up
        print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
        model = setup_architecture(final_params)
        model = maybe_cuda(model, final_params.cuda)
        opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
        agent = agents[final_params.agent](model, opt, final_params)
        print('Training Start')
        x_train_offline = np.concatenate(x_train_offline, axis=0)
        y_train_offline = np.concatenate(y_train_offline, axis=0)
        print("----------run {} training-------------".format(run))
        print('size: {}, {}'.format(x_train_offline.shape, y_train_offline.shape))
        agent.train_learner(x_train_offline, y_train_offline)
        acc_array = agent.evaluate(test_loaders)
        tmp_acc.append(acc_array)



def single_tune_train_val(data_continuum, default_params, tune_params, params_keep, tmp_acc, run):
    tune_data = []
    # prepare val data loader
    test_loaders_full = setup_test_loader(data_continuum.test_data(), default_params)
    tune_test_loaders = test_loaders_full[:default_params.num_val]
    if default_params.online:
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                tune_data.append((x_train, y_train, labels))
                if len(tune_data) == default_params.num_val:
                    # tune
                    best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
                    params_keep.append(best_params)
                    final_params = vars(default_params)
                    final_params.update(best_params)
                    final_params = SimpleNamespace(**final_params)
                    print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
                    break
      ############# run experiment after tuning
        # data_continuum.reset_run()
        # # set up
        # model = setup_architecture(final_params)
        # model = maybe_cuda(model, final_params.cuda)
        # opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
        # agent = agents[final_params.agent](model, opt, final_params)
        # print('Training Start')
        # for i, (x_train, y_train, labels) in enumerate(data_continuum):
        #     print("----------run {} training batch {}-------------".format(run, i))
        #     print('size: {}, {}'.format(x_train.shape, y_train.shape))
        #     agent.train_learner(x_train, y_train)
        #     acc_array = agent.evaluate(test_loaders_full)
        #     tmp_acc.append(acc_array)

    else:
        x_train_offline = []
        y_train_offline = []
        x_tune_offline = []
        y_tune_offline = []
        labels_offline = []
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                x_tune_offline.append(x_train)
                y_tune_offline.append(y_train)
                labels_offline.append(labels)
            x_train_offline.append(x_train)
            y_train_offline.append(y_train)
        tune_data = [(np.concatenate(x_tune_offline, axis=0), np.concatenate(y_tune_offline, axis=0), labels_offline)]
        best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
        params_keep.append(best_params)
        final_params = vars(default_params)
        final_params.update(best_params)
        final_params = SimpleNamespace(**final_params)
        # set up
        print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
        ############# run experiment after tuning
        # model = setup_architecture(final_params)
        # model = maybe_cuda(model, final_params.cuda)
        # opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
        # agent = agents[final_params.agent](model, opt, final_params)
        # print('Training Start')
        # x_train_offline = np.concatenate(x_train_offline, axis=0)
        # y_train_offline = np.concatenate(y_train_offline, axis=0)
        # print("----------run {} training-------------".format(run))
        # print('size: {}, {}'.format(x_train_offline.shape, y_train_offline.shape))
        # agent.train_learner(x_train_offline, y_train_offline)
        # acc_array = agent.evaluate(test_loaders_full)
        # tmp_acc.append(acc_array)


