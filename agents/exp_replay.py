import torch
from torch.utils import data
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.utils import maybe_cuda, AverageMeter
# from RL.RL_replay_base import RL_replay
# from RL.close_loop_cl import close_loop_cl
from torchvision.transforms import transforms

import numpy as np
import torch
import torch.nn as nn
from utils.setup_elements import transforms_match
from utils.utils import cutmix_data


class ExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay, self).__init__(model, opt, params)
        #self.buffer = Buffer(model, params)
        self.buffer = self.memory_manager.buffer
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.softmax_opt =  torch.optim.SGD(self.model.linear.parameters(),
                                lr=0.1,
                                )




        self.replay_para={"mem_ratio":self.params.mem_ratio,
                          "incoming_ratio":self.params.incoming_ratio,
                          "mem_iter":self.params.mem_iters,
                          "randaug_M":self.params.randaug_M}
        self.test_loaders = None


    def _compute_acc(self, batch_x, batch_y,  mem_num=0):


        logits = self.model.forward(batch_x)
        _, pred_label = torch.max(logits, 1)
        acc = (pred_label == batch_y)

        ce_all = torch.nn.CrossEntropyLoss(reduction='none')
        softmax_loss_full = ce_all(logits, batch_y)

        total_num = batch_x.shape[0]
        avrg_acc = acc.sum().item() / total_num
        # loss = torch.mean(softmax_loss_full)

        acc_incoming = acc[mem_num:].sum().item() / (total_num - mem_num)

        incoming_loss = torch.mean(softmax_loss_full[mem_num:])
        self.train_acc_incoming_org.append(acc_incoming)
        self.train_loss_incoming_org.append(incoming_loss.item())
        if(mem_num>0):

            acc_mem = acc[:mem_num].sum().item() / mem_num
            mem_loss = torch.mean(softmax_loss_full[:mem_num])
            self.train_acc_mem_org.append(acc_mem)
            self.train_loss_mem_org.append(mem_loss.item())
            #print(acc_incoming,acc_mem)


    def _batch_update(self,batch_x,batch_y,losses_batch,acc_batch,i,replay_para=None,mem_num=0):
        self.model.train()
        STOP_FLAG = False
        if(replay_para == None):
            replay_para = self.replay_para

        logits = self.model.forward(batch_x)
        _, pred_label = torch.max(logits, 1)
        acc = (pred_label == batch_y)

        ce_all = torch.nn.CrossEntropyLoss(reduction='none')
        softmax_loss_full = ce_all(logits, batch_y)

        total_num = batch_x.shape[0]
        avrg_acc = acc.sum().item() / total_num
        #loss = torch.mean(softmax_loss_full)






        acc_incoming = acc[mem_num:].sum().item() / (total_num - mem_num)

        incoming_loss = torch.mean(softmax_loss_full[mem_num:])
        self.train_loss_incoming.append(incoming_loss.item())
        self.train_acc_incoming.append(acc_incoming)

        if(mem_num>0):

            acc_mem = acc[:mem_num].sum().item() / mem_num
            mem_loss = torch.mean(softmax_loss_full[:mem_num])
            self.train_acc_mem.append(acc_mem)
            self.train_loss_mem.append(mem_loss.item())
            # if(self.close_loop_cl != None):### used for state construction
            #
            #     self.close_loop_cl.last_train_loss = mem_loss.item()


            #loss = mem_loss+ incoming_loss
            loss = replay_para['mem_ratio'] * mem_loss+ \
                   replay_para['incoming_ratio'] * incoming_loss
            #loss = torch.mean(softmax_loss_full)
            train_stats = {'acc_incoming': acc_incoming,
                           'acc_mem': acc_mem,
                           "loss_incoming": incoming_loss.item(),
                           "loss_mem": mem_loss.item(),
                           "batch_num": i,
                           }

        else:
            #loss = torch.mean(softmax_loss_full)
            loss = replay_para['incoming_ratio'] * incoming_loss
            #loss = incoming_loss
            acc_mem = None
            mem_loss = None
            train_stats=None


        acc_batch.update(avrg_acc, batch_y.size(0))
        losses_batch.update(loss.item(), batch_y.size(0))

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.loss_batch.append(loss.item())



        return  train_stats

    def early_stop_check(self,train_stats):
        if(train_stats == None):
            return False
        mem_loss=train_stats["loss_mem"]
        incoming_loss = train_stats["loss_incoming"]
        acc_mem = train_stats["acc_mem"]
        STOP_FLAG= False
        if (self.params.dyna_mem_iter in ["STOP_loss", "STOP_acc_loss"]):

            if (mem_loss > incoming_loss * self.params.stop_ratio):
                STOP_FLAG = True
        elif (self.params.dyna_mem_iter in ["STOP_acc", "STOP_acc_loss"]):
            if (acc_mem > self.params.train_acc_max):
                STOP_FLAG = True
        else:
            STOP_FLAG = False
        return STOP_FLAG
    def immediate_evaluate(self):
        if(self.params.immediate_evaluate == False):
            return None, None
        if(self.test_loaders == None):
            return None, None
        #acc_array = self.evaluate(self.test_loaders,no_nmc=True)
        acc_array,loss_array = self.softmax_evaluate(self.test_loaders)
        return acc_array,loss_array
    def test_add_buffer(self,inc_batch_x,inc_batch_y):
        task_seen = self.task_seen
        mem_batch_x,mem_batch_y = self.memory_manager.retrieve_from_main(inc_batch_x,inc_batch_y,retrieve_num = 100)
        mem_batch_x_add,mem_batch_y_add = self.memory_manager.retrieve_from_add(inc_batch_x,inc_batch_y,retrieve_num=100)
        if(mem_batch_x.shape[0]>0):


            with torch.no_grad():
                logits = self.model.forward(mem_batch_x)
                _, pred_label = torch.max(logits, 1)
                acc = (pred_label == mem_batch_y)

                ce_all = torch.nn.CrossEntropyLoss(reduction='none')
                softmax_loss_full = ce_all(logits, mem_batch_y)

                total_num = mem_batch_x.shape[0]
                avrg_acc_main = acc.sum().item() / total_num
                self.train_acc_mem_main.append(avrg_acc_main)
        else:
            avrg_acc_main = 0


        if(mem_batch_x_add.shape[0]>0):

            with torch.no_grad():
                logits = self.model.forward(mem_batch_x_add)
                _, pred_label = torch.max(logits, 1)
                acc = (pred_label == mem_batch_y_add)

                ce_all = torch.nn.CrossEntropyLoss(reduction='none')
                softmax_loss_full = ce_all(logits, mem_batch_y_add)

                total_num = mem_batch_x_add.shape[0]
                avrg_acc_add = acc.sum().item() / total_num
                self.train_acc_mem_add.append(avrg_acc_add)
        else:
            avrg_acc_add = 0
        return avrg_acc_main, avrg_acc_add


    def train_learner(self, x_train, y_train):


        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=4,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()
        acc_batch = AverageMeter()
        acc_mem = AverageMeter()
        test_acc_list=[]
        acc_main=None
        acc_add = None

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update

                batch_x,batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                self.set_memIter()
                memiter=self.mem_iters

                for j in range(self.mem_iters):



                    concat_batch_x,concat_batch_y,mem_num = self.concat_memory_batch(batch_x,batch_y)




                    stats = self._batch_update(concat_batch_x,concat_batch_y, losses_batch, acc_batch, i,mem_num=mem_num)
                    STOP_FLAG = self.early_stop_check(stats)
                    if(STOP_FLAG ):
                        memiter=j+1
                        break
                    test_acc,test_loss = self.immediate_evaluate()
                    test_acc_list.append(test_acc)
                    self.test_acc_true.append(test_acc)
                    self.test_loss_true.append(test_loss)

                    if(self.params.test_add_buffer == True):
                        acc_main, acc_add = self.test_add_buffer(batch_x,batch_y)


                self.mem_iter_list.append(memiter)

                self.memory_manager.update_memory(batch_x, batch_y)

                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )

        self.after_train()
        return test_acc_list


