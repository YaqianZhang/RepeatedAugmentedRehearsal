from utils.buffer.buffer import Buffer
from utils.buffer.tmp_buffer import Tmp_Buffer
from utils.buffer.test_buffer import Test_Buffer
import torch
import numpy as np

class memory_manager_class(object):
    def __init__(self,model,params):
        self.params = params
        self.model = model
        self.buff_use = "single buff"

        ## main buffer
        self.buffer = Buffer(model, params)
        self.current_performance = []

        # ## additional buffer
        # if (self.params.switch_buffer_type in [ "two_buffer","dyna_buffer"]):
        #     self.buffer_add = Buffer(model, params)
        # self.buff_use ="main buff"
        self.buff_replay_times = 0

        ## test buffer
        if (params.RL_type != "NoRL"  or params.use_test_buffer == True):
           #self.test_buffer = Test_Buffer(params, )
            self.build_test_buffer()
            self.params.use_test_buffer = True
        #
        #
        # ## tmp buffer
        # if(params.use_tmp_buffer):
        #     self.tmp_buffer=Tmp_Buffer(model,params,self.buffer)
        # else:
        #     self.tmp_buffer = None

        self.tmp_buffer = None
    def build_test_buffer(self):
        if(self.params.test_buffer_type == "class_balance"):
            self.test_buffer = Test_Buffer(self.params, )
        else:
            self.test_buffer = Buffer(self.model, self.params,mem_size=self.params.test_mem_size)


    # def update_task_number(self):
    #     self.buffer.task_seen_so_far += 1
    # def test_memory_ready(self):
    #     return self.test_buffer.current_index >= 50  # self.params.test_mem_batchSize
    #
    #
    # def reset_buffer(self, params):
    #     if (self.params.RL_type != "NoRL"):
    #         self.build_test_buffer()
    #         #self.test_buffer = Test_Buffer(params, )
    #     self.buffer = Buffer(self.model, params)
    #     self.buffer_add = Buffer(self.model, params)
    # def compute_incoming_influence(self,incoming_x,incoming_y):
    #     return self.buffer.compute_incoming_influence(x=incoming_x,y=incoming_y)
    #

    def retrieve_from_mem(self,batch_x, batch_y,task_seen,retrieve_num=None):

        mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y,retrieve_num=retrieve_num)
        # if (self.params.switch_buffer_type == "one_buffer"):
        #     mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
        # elif (self.params.switch_buffer_type == "two_buffer"):
        #
        #     if (task_seen % 2 == 1):
        #         mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
        #         self.buff_use = "main buff"
        #
        #
        #     else:
        #         mem_x, mem_y = self.buffer_add.retrieve(x=batch_x, y=batch_y)
        #         self.buff_use = "2nd buff"
        #         #mem_x, mem_y = random_retrieve(self.test_buffer, 10)
        # elif (self.params.switch_buffer_type == "dyna_buffer"):
        #     if(self.buff_use == "main buff"):
        #         mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
        #     else:
        #         mem_x, mem_y = self.buffer_add.retrieve(x=batch_x, y=batch_y)
        #     self.buff_replay_times += 1
        #
        #     if(self.buff_replay_times>self.params.switch_buffer_freq):
        #         #switch to test buffer
        #
        #         self.buff_replay_times=0
        #         if(self.buff_use=="main buff"):
        #             self.buff_use = "2nd buff"
        #         else:
        #             self.buff_use = "main buff"
        #
        # else:
        #     raise NotImplementedError("Undefined buffer switch strategy", self.params.switch_buffer_type)
        return mem_x, mem_y


    # def update_before_training(self, batch_x, batch_y):
    #
    #     ## update test memory
    #     if (self.params.test_mem_type == "after"):
    #         return batch_x, batch_y
    #
    #     test_size = int(batch_x.shape[0] * 0.1)
    #     # print("save batch to test buffer and buffer",test_size)
    #     self.test_buffer.update(batch_x[:test_size], batch_y[:test_size])
    #     print("update test memory")
    #
    #     return batch_x[test_size:], batch_y[test_size:]
    # def reset_new_old(self):
    #     self.buffer.buffer_new_old = torch.LongTensor(self.buffer.buffer_size).fill_(0)

    def update_memory(self, batch_x, batch_y):

        #self.buffer.update(batch_x, batch_y, )

        # save some parts of batch_x and batch_y into the memory
        if (self.params.use_test_buffer and self.params.test_mem_type == "after"):
            test_size = int(batch_x.shape[0] * 0.1)
            # print("save batch to test buffer and buffer",test_size)
            self.test_buffer.update(batch_x[:test_size], batch_y[:test_size])

            # if (self.params.switch_buffer_type == "two_buffer" or self.params.switch_buffer_type == "dyna_buffer"):
            #
            #     buffer_size = int(batch_x.shape[0] * 0.55)
            #
            #     self.buffer.update(batch_x[test_size:buffer_size], batch_y[test_size:buffer_size], self.tmp_buffer)
            #     self.buffer_add.update(batch_x[buffer_size:], batch_y[buffer_size:], self.tmp_buffer)
            # else:
            self.buffer.update(batch_x[test_size:], batch_y[test_size:], self.tmp_buffer)

        else:
            # if (self.params.switch_buffer_type == "two_buffer" or self.params.switch_buffer_type == "dyna_buffer"):
            #
            #     buffer_size = int(batch_x.shape[0] * 0.5)
            #
            #     self.buffer.update(batch_x[0:buffer_size], batch_y[0:buffer_size], self.tmp_buffer)
            #     self.buffer_add.update(batch_x[buffer_size:], batch_y[buffer_size:], self.tmp_buffer)
            # else:
            self.buffer.update(batch_x, batch_y, self.tmp_buffer)



