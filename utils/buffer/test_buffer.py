from utils.setup_elements import input_size_match
from utils import name_match  # import update_methods, retrieve_methods
from utils.utils import maybe_cuda
import torch
import random
import numpy as np
from utils.buffer.recycle import recycle




class Test_Buffer(torch.nn.Module):
    def __init__(self,  params,):
        super().__init__()
        self.params = params
        #self.model = model
        self.cuda = self.params.cuda
        self.current_index = 0
        self.n_seen_so_far = 0
        self.training_steps = 0

        # define buffer
        self.buffer_size = params.test_mem_size
        print('test buffer has %d slots' % self.buffer_size)
        self.input_size = input_size_match[params.data]
        print("test buffer"+str(self.buffer_size))

        # self.buffer_img = torch.FloatTensor(buffer_size, *input_size).fill_(0)
        # self.buffer_label = torch.LongTensor(buffer_size).fill_(0)

        self.mem_img = {}
        self.mem_c = {}

        # define update and retrieve method

        self.update_method = name_match.update_methods[params.update](params,)
        self.retrieve_method = name_match.retrieve_methods[params.retrieve](params)

        if (self.params.test_mem_recycle):
            self.recycler = recycle()

    # def update(self, x, y,tmp_buffer=None):
    #     return self.update_method.update(buffer=self, x=x, y=y,tmp_buffer=tmp_buffer)

    def update(self, x, y,tmp_buffer=None):
        for i in range(len(y)):
            self.greedy_balancing_update(x[i],y[i].item())
        index = 0
        for k in self.mem_c:
            index += self.mem_c[k]
        self.current_index = index

    def retrieve_strict_blc(self):

        min_class = min(self.mem_c.keys(), key=(lambda k: self.mem_c[k]))

        num_per_class = self.mem_c[min_class]
        mem_x = []
        mem_y = []
        for i in self.mem_img.keys():
            perm_idx = np.array(np.random.permutation(len(self.mem_img[i])))
            perm_arr = [self.mem_img[i][k] for k in perm_idx]
            selected_img = perm_arr[:num_per_class]

            mem_x += selected_img
            mem_y += [i] * num_per_class #self.mem_c[i]

        mem_x = torch.stack(mem_x)
        mem_y = torch.LongTensor(mem_y)
        mem_x = maybe_cuda(mem_x)
        mem_y = maybe_cuda(mem_y)
        return mem_x,mem_y
        #



    def retrieve(self,retrieve_num=None):
        mem_x = []
        mem_y = []
        for i in self.mem_img.keys():
            mem_x += self.mem_img[i]
            mem_y += [i] * self.mem_c[i]

        mem_x = torch.stack(mem_x)
        mem_y = torch.LongTensor(mem_y)
        mem_x = maybe_cuda(mem_x)
        mem_y = maybe_cuda(mem_y)


        return mem_x,mem_y

    # def overwrite(self,idx_map,x,y):
    #
    #     self.buffer_img[list(idx_map.keys())] = x[list(idx_map.values())]
    #     self.buffer_label[list(idx_map.keys())] = y[list(idx_map.values())]
    #
    # def reset(self):
    #     buffer_size = self.params.mem_size
    #     print('buffer has %d slots' % buffer_size)
    #     input_size = input_size_match[self.params.data]
    #     self.buffer_img = torch.FloatTensor(buffer_size, *input_size).fill_(0)
    #     self.buffer_label = torch.LongTensor(buffer_size).fill_(0)
    #     self.buffer_replay_times =maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
    #     self.buffer_last_replay = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))


    def greedy_balancing_update(self, x, y):
        k_c = self.buffer_size // max(1, len(self.mem_img))
        if y not in self.mem_img or self.mem_c[y] < k_c:
            if sum(self.mem_c.values()) >= self.buffer_size:
                cls_max = max(self.mem_c.items(), key=lambda k:k[1])[0]
                idx = random.randrange(self.mem_c[cls_max])
                img = self.mem_img[cls_max].pop(idx)
                if(self.params.test_mem_recycle):
                    recycle.store_tmp(img,cls_max)
                self.mem_c[cls_max] -= 1
            if y not in self.mem_img:
                self.mem_img[y] = []
                self.mem_c[y] = 0
            self.mem_img[y].append(x)
            self.mem_c[y] += 1



