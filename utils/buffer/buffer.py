from utils.setup_elements import input_size_match
from utils import name_match #import update_methods, retrieve_methods
from utils.utils import maybe_cuda
import torch
import numpy as np
from utils.buffer.buffer_utils import BufferClassTracker
from utils.setup_elements import n_classes

class Buffer(torch.nn.Module):
    def __init__(self, model, params,mem_size=None,RL_agent=None, RL_env=None,):
        super().__init__()
        self.params = params
        self.model = model
        self.cuda = self.params.cuda
        self.current_index = 0
        self.n_seen_so_far = 0
        self.buffer_used_steps = 0
        self.device = "cuda" if self.params.cuda else "cpu"

        if(mem_size==None):
            mem_size = self.params.mem_size

        # define buffer
        buffer_size =mem_size
        self.buffer_size = mem_size
        print('buffer has %d slots' % buffer_size)
        input_size = input_size_match[params.data]
        self.buffer_img = torch.FloatTensor(buffer_size, *input_size).fill_(0)
        self.buffer_label = torch.LongTensor(buffer_size).fill_(0)
        self.buffer_new_old = torch.LongTensor(buffer_size).fill_(0)
        #self.buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        #self.buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        self.buffer_replay_times =maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        self.buffer_last_replay = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        self.unique_replay_list=[]
        self.replay_sample_label=[]

        # registering as buffer allows us to save the object using `torch.save`
        #self.register_buffer('buffer_img', buffer_img)
        #self.register_buffer('buffer_label', buffer_label)

        # define update and retrieve method
        self.update_method = name_match.update_methods[params.update](params)
        if(params.retrieve == "RL"):
            self.retrieve_method = name_match.retrieve_methods[params.retrieve](params,RL_agent, RL_env)
        else:
            self.retrieve_method = name_match.retrieve_methods[params.retrieve](params)
        self.random_retrieve_method = name_match.retrieve_methods["random"](params)

    def update_replay_times(self, indices):
        self.buffer_replay_times[indices]+=1
        self.buffer_last_replay +=1
        self.buffer_last_replay[indices] =0
    def reset(self):
        buffer_size = self.params.mem_size
        print('buffer has %d slots' % buffer_size)
        input_size = input_size_match[self.params.data]
        self.buffer_img = torch.FloatTensor(buffer_size, *input_size).fill_(0)
        self.buffer_label = torch.LongTensor(buffer_size).fill_(0)
        self.buffer_replay_times =maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        self.buffer_last_replay = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        self.current_index = 0
        self.n_seen_so_far = 0

        self.buffer_used_steps = 0


    def update(self, x, y,tmp_buffer=None):
        self.buffer_used_steps += 1
        return self.update_method.update(buffer=self, x=x, y=y,tmp_buffer=tmp_buffer)

    def compute_incoming_influence(self, **kwargs):
        return self.retrieve_method.compute_incoming_influence(buffer=self, **kwargs)

    def retrieve_class_num(self,class_num_list,num_retrieve):
        valid_each_class = []
        for c in class_num_list:
            valid_each_class.append(self.buffer_label[:self.current_index].numpy() == c )
        valid_each_class = np.array(valid_each_class)
        boolean_indices =  np.sum(valid_each_class,axis = 0)>=1

        valid_indices = np.arange(self.current_index)[boolean_indices]
        #assert False


        indices = torch.from_numpy(np.random.choice(valid_indices, num_retrieve, )).long()

        x = self.buffer_img[indices]

        y = self.buffer_label[indices]

        x = maybe_cuda(x)
        y = maybe_cuda(y)
        return x,y

    def retrieve_class_balance_sample(self,num_retrieve):
        labels = torch.unique(self.buffer_label[:self.current_index])
        ##### method 1
        num_per_class = num_retrieve // len(labels)
        # ##### method 2
        # num_of_labels = []
        # for label in labels:
        #     idx = self.buffer_label[:self.current_index] == label
        #     num_of_labels.append(torch.sum(idx).item())
        # valid_num = np.min(num_of_labels)
        # num_per_class = valid_num
        #print("!!!numperclass",valid_num,num_of_labels,labels)
       # assert False
        selected_imgs = []
        selected_labels = []
        for label in labels:
            idx = self.buffer_label[:self.current_index] == label
            valid_img = self.buffer_img[:self.current_index][idx]
            perm_idx = np.random.permutation(len(valid_img)).tolist()
            selected_img = valid_img[perm_idx][:num_per_class]
            selected_imgs.append(selected_img)
            selected_label = self.buffer_label[:self.current_index][idx][perm_idx][:num_per_class]
            selected_labels.append(selected_label)
        selected_imgs = torch.cat(selected_imgs,dim=0)
        selected_labels = torch.cat(selected_labels,dim=0)
        return selected_imgs,selected_labels




    def random_retrieve(self,**kwargs):
        return self.random_retrieve_method.retrieve(buffer=self, **kwargs)

    def retrieve(self, **kwargs):
        # if(self.retrieve_method.num_retrieve==-1):
        #     print("dynamic mem batch size")
        #
        #     self.retrieve_method.num_retrieve = self.task_seen_so_far * 10 # to-do: change 10 to the batch size of new data
        return self.retrieve_method.retrieve(buffer=self, **kwargs)
    def save_buffer_info(self,prefix=""):
        removed_sample = np.array(self.unique_replay_list)
        arr = self.buffer_replay_times.detach().cpu().numpy()
        np.save(prefix+"_removed_sample.npy",removed_sample)
        np.save(prefix+"_remain_sample.npy",arr)
        np.save(prefix+"_sample_label.npy", np.array(self.replay_sample_label))
        np.save(prefix+"_sample_label_remain.npy", self.buffer_label.detach().cpu().numpy())

    def overwrite(self,idx_map,x,y):
        ## zyq: save replay_times
        #print("----buffer overwrite")
        for i in list(idx_map.keys()):
            replay_times = self.buffer_replay_times[i].detach().cpu().numpy()
            self.unique_replay_list.append(int(replay_times))
            self.buffer_replay_times[i]=0
            self.buffer_last_replay[i]=0
            sample_label = int(self.buffer_label[i].detach().cpu().numpy())
            self.replay_sample_label.append(sample_label)

        self.buffer_img[list(idx_map.keys())] = x[list(idx_map.values())]
        self.buffer_label[list(idx_map.keys())] = y[list(idx_map.values())]
        self.buffer_new_old[list(idx_map.keys())]=1


