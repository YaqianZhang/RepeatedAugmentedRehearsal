from torch.utils import data
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
from agents.exp_replay import  ExperienceReplay

from scipy.stats import linregress

# from RL.RL_replay_base import RL_replay
#
# from RL.close_loop_cl import close_loop_cl

import numpy as np


def softmax(vec, j=0):
    # nn=vec-np.mean(vec)

    para = 0  # 1.0/ np.sqrt(j+1)

    noise = np.random.rand(1) * 2 - 1  ###[-1, 1]
    nn = (1 - para) * vec + (para) * noise

    nn = nn - np.max(nn)
    # print np.max(nn)

    nn1 = np.exp(nn)
    # print np.max(nn1)
    # nn1 = 1/(1+np.exp(-nn))
    vec_prob = nn1 * 1.0 / np.sum(nn1)
    return vec_prob


class ER_dyna_iter_aug_dbpg_joint(ExperienceReplay):
    def __init__(self, model, opt, params):
        super(ER_dyna_iter_aug_dbpg_joint, self).__init__(model, opt, params)
        #if(self.params.online_hyper_RL or self.params.scr_memIter ):

        # self.close_loop_cl = close_loop_cl(self,model, self.memory_manager)
        #
        # self.RL_replay = RL_replay(params,self.close_loop_cl)
        if(params.aug_action_num == 8):
            self.aug_map=[(1,5),(1,14),(2,5),(2,14),(3,5),(3,14),(4,5),(4,14)]
        elif(params.aug_action_num == 10):
            self.aug_map=[(1,5),(1,14),(2,5),(2,14),(3,5),(3,14),(4,5),(4,14),(5,14),(6,14)]
        else:
            self.aug_map = [(1, 5), (1, 14),  (2, 14), (3, 14),  (4, 14)]
        self.aug_map.reverse()
        self.iter_map=[1,5,10,15,20]
        self.iter_map= list(np.arange(1,21))
        # self.action_map = []
        # for aug in aug_map:
        #     for iters in iter_map:
        #         action = list(aug) + [iters]
        #         self.action_map.append(action)
        self.action_num_iter =len(self.iter_map)
        self.weights_iter = 100*np.ones(self.action_num_iter)
        self.action_prob_iter = softmax(self.weights_iter)
        self.action_num_aug =len(self.aug_map)
        self.weights_aug = 100*np.ones(self.action_num_aug)
        self.action_prob_aug = softmax(self.weights_aug)



        self.mem_iter_min = self.params.mem_iter_min
        self.mem_iter_max = self.params.mem_iter_max
        self.reward_list=[]
        for i in range(self.action_num_iter):
            self.reward_list.append([0.5])
        self.N=None



    def decrease_iter_condition(self,last_acc,target_acc_end):
        return last_acc <target_acc_end


    def drift_detection(self):
        reward_mean=[]
        recent = 20
        for i in range(self.action_num_iter):
            avrg = np.mean(self.reward_list[i][-recent:])
            reward_mean.append(avrg)
        current_opt=np.argmax(self.weights)
        reward_opt=np.max(reward_mean)
        current_max = reward_mean[current_opt]
        if(reward_opt - current_max > 0.3):
            return True
        else:
            return False
    def update_reward(self,reward,action):
        self.reward_list[action].append(reward)




    def update_weight_bpg(self,train_acc_list,STOP_FLAG,action_id,weights,prob,target_acc_start,target_acc_end):
        current_iter = len(train_acc_list)
        if(current_iter ==0 or current_iter == None):
            return
        # #prob = self.action_prob
        last_acc = train_acc_list[-1]


        alpha =  5
        alpha_too_large = self.params.bpg_lr_large
        alpha_too_small = self.params.bpg_lr_small
        #action_id = current_iter -self.mem_iter_min

        if(last_acc>target_acc_end  ): ## too large or STOP_FLAG
           # print("too large",last_acc,target_acc_end)
            #logs=[current_iter,"too large",last_acc,target_acc_end,STOP_FLAG]

            if np.sum(prob[action_id:])>0:
                weights[action_id:] -= alpha_too_large * prob[action_id]/np.sum(prob[action_id:])


            if np.sum(prob[:action_id+1])>0:
                weights[:action_id] +=alpha_too_large * prob[action_id]/np.sum(prob[:action_id+1])
        elif(last_acc <target_acc_start ): ## too small reduce the smaller
            #print("too small",last_acc,target_acc_start)
            #logs=[current_iter,"too small", last_acc,target_acc_start]
            ## better
            if np.sum(prob[action_id:])>0:
                weights[action_id:] += alpha_too_small * prob[action_id]/np.sum(prob[action_id:])

            ## worse
            if np.sum(prob[:action_id+1])>0:
                weights[:action_id] -=alpha_too_small * prob[action_id]/np.sum(prob[:action_id+1])

        else:
            #logs =[current_iter,prob]
            pass
        prob[:] = softmax(weights) # 200*1
        return None
    def restart_bpg(self):


        self.weights_iter = 100 * np.ones(self.action_num_iter)
        self.action_prob_iter = softmax(self.weights_iter)

        self.weights_aug = 100 * np.ones(self.action_num_aug)
        self.action_prob_aug = softmax(self.weights_aug)

    def sample_action_bpg_aug(self):

        ### sample from the action probabilty
        #self.action_prob = softmax(self.weights) # 200*1
        # print(self.action_num,self.action_prob.shape)
        # assert False



        action_idx = np.random.choice(range(0,self.action_num_aug), 1, replace=False, p=self.action_prob_aug)[0]

        return self.aug_map[action_idx],action_idx ## choice [action]
    def sample_action_bpg(self):

        ### sample from the action probabilty
        #self.action_prob = softmax(self.weights) # 200*1
        # print(self.action_num,self.action_prob.shape)
        # assert False



        action_idx = np.random.choice(range(0,self.action_num_iter), 1, replace=False, p=self.action_prob_iter)[0]

        return self.iter_map[action_idx],action_idx ## choice [action]
    def train_learner(self, x_train, y_train):
        if(self.params.bpg_restart):
            self.restart_bpg()
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()
        acc_batch = AverageMeter()
        acc_mem = AverageMeter()
        replay_para = None
        STOP_FLAG = False
        target_acc_iter_min = self.params.train_acc_min #0.80
        target_acc_iter_max =self.params.train_acc_max #0.9
        target_acc_aug_min = self.params.train_acc_min_aug #0.80
        target_acc_aug_max =self.params.train_acc_max_aug #0.
        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):

                batch_x,batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                #batch_x,batch_y = self.memory_manager.update_before_training(batch_x,batch_y)
                memiter,action_iter = self.sample_action_bpg()
                aug_para,action_aug= self.sample_action_bpg_aug()
                [N,M]=aug_para
                #self.mem_iter_list.append(memiter)
                #self.RL_replay.RL_agent.greedy_action.append(memiter)
                aug_strength=N*100+M
                self.aug_agent.set_aug_NM(N, M)
                self.aug_N_list.append(aug_strength)
                train_acc_list = []
                train_loss_list=[]
                DETECT = False


                for j in range(memiter):


                    concat_batch_x, concat_batch_y, mem_num = self.concat_memory_batch(batch_x, batch_y)

                    train_stats = self._batch_update(concat_batch_x, concat_batch_y, losses_batch, acc_batch, i,replay_para,mem_num=mem_num)
                    if(train_stats != None):
                        train_acc_list.append(train_stats['acc_mem'])
                        train_loss_list.append(train_stats['loss_mem'])
                    STOP_FLAG = self.early_stop_check(train_stats)
                    # if(STOP_FLAG ):
                    #     memiter=j+1
                    #     break

                self.mem_iter_list.append(memiter)
                logs = self.update_weight_bpg(train_acc_list,STOP_FLAG,action_iter,self.weights_iter,self.action_prob_iter,
                                              target_acc_iter_min,target_acc_iter_max)
                logs = self.update_weight_bpg(train_acc_list,True,action_aug,self.weights_aug,self.action_prob_aug,
                                              target_acc_aug_min,target_acc_aug_max)

                if(train_stats!= None):
                    reward = np.abs(train_stats['acc_mem']-0.9)
                    self.update_reward(reward,action_iter)
                    if(self.params.drift_detection):
                        DETECT = self.drift_detection()
                        if(DETECT):
                            self.restart_bpg()
                # if(train_stats != None):
                #     N=self.adjust_aug(train_stats)
                #     self.aug_N_list.append(N)
                # else:
                #     N=0

                self.memory_manager.current_performance=train_acc_list
                self.restart_flag.append(DETECT)


                self.memory_manager.update_memory(batch_x, batch_y)

                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )
                    print(
                        '==>>> it: {}, mem avg. loss: {:.6f}, '
                        'running mem acc: {:.3f}'
                            .format(i, losses_mem.avg(), acc_mem.avg())
                    )
                    print("memiter",memiter,"aug",aug_strength,"mem_acc",train_acc_list[-1])#np.max(train_acc_list),train_acc_list[-1],np.mean(train_acc_list))

                    #print("replay_para", replay_para,"action:",self.RL_replay.RL_agent.greedy,self.RL_replay.action,)
        self.after_train()

