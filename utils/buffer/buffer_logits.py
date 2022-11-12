from utils.setup_elements import input_size_match
from utils import name_match #import update_methods, retrieve_methods
from utils.utils import maybe_cuda
import torch
import numpy as np
from utils.buffer.buffer_utils import BufferClassTracker
from utils.setup_elements import n_classes
from utils.buffer.buffer import Buffer

class Buffer_logits(Buffer):
    def __init__(self, model, params,mem_size=None,RL_agent=None, RL_env=None,):
        super().__init__(model,params,)
        if(mem_size==None):
            mem_size = self.params.mem_size
        buffer_size =mem_size
        self.buffer_size = mem_size
        print('buffer has %d slots' % buffer_size)
        input_size = input_size_match[params.data]
        class_num = n_classes[self.params.data]
        self.buffer_logits = torch.FloatTensor(buffer_size, class_num).fill_(0)
        self.buffer_label = torch.LongTensor(buffer_size).fill_(0)



    def update(self, x, y,logits=None,tmp_buffer=None):
        self.buffer_used_steps += 1
        return self.update_method.update(buffer=self, x=x, y=y,logits=logits,tmp_buffer=tmp_buffer)

    def retrieve(self, **kwargs):
        # if(self.retrieve_method.num_retrieve==-1):
        #     print("dynamic mem batch size")
        #
        #     self.retrieve_method.num_retrieve = self.task_seen_so_far * 10 # to-do: change 10 to the batch size of new data
        return self.retrieve_method.retrieve(buffer=self, **kwargs)
    def overwrite(self,idx_map,x,y,logits):
        ## zyq: save replay_times
        #print("----buffer overwrite")
        # for i in list(idx_map.keys()):
        #     replay_times = self.buffer_replay_times[i].detach().cpu().numpy()
        #     self.unique_replay_list.append(int(replay_times))
        #     self.buffer_replay_times[i]=0
        #     self.buffer_last_replay[i]=0
        #     sample_label = int(self.buffer_label[i].detach().cpu().numpy())
        #     self.replay_sample_label.append(sample_label)

        self.buffer_img[list(idx_map.keys())] = x[list(idx_map.values())]
        self.buffer_label[list(idx_map.keys())] = y[list(idx_map.values())]
        self.buffer_logits[list(idx_map.keys())] = logits[list(idx_map.values())]
        self.buffer_new_old[list(idx_map.keys())]=1


