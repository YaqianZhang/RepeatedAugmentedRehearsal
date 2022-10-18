from utils.setup_elements import input_size_match
from utils import name_match  # import update_methods, retrieve_methods
from utils.utils import maybe_cuda
import torch
import numpy as np



class Tmp_Buffer(torch.nn.Module):
    def __init__(self, model, params,buffer):
        super().__init__()
        self.buffer= buffer
        self.params = params
        self.model = model
        input_size = input_size_match[params.data]
        self.tmp_buffer_size = params.mem_size   # to-do: zyq set tmp buffer size
        self.tmp_x = maybe_cuda(torch.FloatTensor(self.tmp_buffer_size, *input_size).fill_(0))
        self.tmp_y = maybe_cuda(torch.LongTensor(self.tmp_buffer_size).fill_(0))
        self.current_n = 0



        # define buffer




    def tmp_store(self,batch_x,batch_y):

        new_num = batch_x.size(0)
        self.tmp_x[self.current_n:self.current_n+new_num]=batch_x
        self.tmp_y[self.current_n:self.current_n+new_num]=batch_y
        self.current_n += new_num
        #print("tmp store",self.tmp_x.size(0),self.current_n,new_num,batch_x.size(0),batch_y.size(0))

    def reset(self):
        self.tmp_x = torch.zeros_like(self.tmp_x)
        self.tmp_y = torch.zeros_like(self.tmp_y)
        print("reset tmp memory, space used in tmp memory",self.current_n)
        self.current_n = 0




    def update_true_buffer(self):
        #idx_buffer = torch.FloatTensor(self.current_n).to(self.tmp_x.device).uniform_(0, self.params.mem_size).long()
        idx_buffer = torch.FloatTensor(self.current_n).uniform_(0, self.params.mem_size).long()

        idx_new_data = torch.range(0,self.current_n)

        idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(idx_buffer.size(0))}
        print(self.tmp_x.device,self.tmp_y.device,idx_buffer.device,idx_new_data.device)
        self.buffer.overwrite(idx_map, self.tmp_x.cpu(), self.tmp_y.cpu())
        # print("to be store",self.tmp_y[self.current_n])
        # print("buffer indices",idx_buffer)
        # print("after replacement",self.buffer.buffer_label[idx_buffer])

        self.reset()
