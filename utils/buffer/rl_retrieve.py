import torch
from utils.buffer.buffer_utils import random_retrieve

import copy


class RL_retrieve(object):
    def __init__(self, params,RL_agent,RL_env):

        super().__init__()
        self.RL_agent = RL_agent
        self.RL_env = RL_env

        self.params = params
        self.subsample = params.subsample
        self.num_retrieve = params.eps_mem_batch

    def retrieve(self, buffer, **kwargs):
        sub_x, sub_y, mem_indices = random_retrieve(buffer, self.subsample, return_indices=True)
        #sub_x, sub_y = random_retrieve(buffer, self.subsample)



        if sub_x.size(0) > 0:

            ## TODO-zyq: rl
            # mem_indices [0,5000], big_ind [0,50]
            #state = env.compute_state(mem_indices) # 50*3
            action = self.RL_agent.sample_action()
            big_ind = self.RL_env.from_action_to_indices(action,buffer,mem_indices)
            buffer.update_replay_times(mem_indices[big_ind])
            #print("RL retreive",)
            return sub_x[big_ind], sub_y[big_ind]
        else:
            return sub_x, sub_y

        #
        # if sub_x.size(0) > 0:
        #     ## TODO-zyq: rl
        #     # mem_indices [0,5000], big_ind [0,50]
        #     state = env.compute_state(mem_indices) # 50*3
        #     action = RL_agent.sample(state)
        #     big_ind =  env.intepret(action)
        #     reward, next_state = env.step(action,)
        #     RL_agent.update_agent(state, action, reward, next_state)  # todo
        #     buffer.update_replay_times(mem_indices[big_ind])
        #     return sub_x[big_ind], sub_y[big_ind]
        # else:
        #     return sub_x, sub_y

