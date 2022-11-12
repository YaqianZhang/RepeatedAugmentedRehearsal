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
from agents.exp_replay import ExperienceReplay
import time


class DER(ExperienceReplay):
    def __init__(self, model, opt, params):
        super(DER, self).__init__(model, opt, params)

        self.buffer = self.memory_manager.buffer
        self.alpha = params.DER_alpha #0.3




    def aug_data(self,batch_x,):
        if (self.task_seen >= self.params.aug_start):
            if (self.params.randaug):
                # print(concat_batch_x[0])

                # batch_x_aug1 = self.aug_agent.aug_data(batch_x,mem_x.size(0),)
                # end1=time.time()

                batch_x_aug2 = self.aug_agent.aug_data_old_batch(batch_x )
                batch_x = batch_x_aug2
            elif (self.params.deraug):
                self.aug_agent.set_deraug()
                batch_x_aug2 = self.aug_agent.aug_data_old_batch(batch_x)
                batch_x = batch_x_aug2
            # elif (self.params.scraug):
            #     batch_x = self.aug_agent.scr_aug_data(batch_x)
            else:
                pass

            if (self.params.aug_normal):
                # print(batch_x)
                # assert False

                batch_x = self.transform_normal(batch_x)
            return batch_x

    def train_learner(self, x_train, y_train): ## todo: move build of dataloader from agent to dataset script


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
                #batch_x,batch_y = self.memory_manager.update_before_training(batch_x,batch_y)
                self.set_memIter()
                memiter=self.mem_iters
                start_time = time.time()

                for j in range(self.mem_iters):
                    self.opt.zero_grad()




                    aug_batch_x = self.aug_data(batch_x)
                    aug_inc_logits = self.model.forward(aug_batch_x)

                    # ce_all = torch.nn.CrossEntropyLoss(reduction='none')
                    # softmax_loss_full = ce_all(aug_inc_logits, batch_y)
                    # incoming_loss = torch.mean(softmax_loss_full)

                    ce_mean = torch.nn.CrossEntropyLoss(reduction='mean')
                    der_loss = ce_mean(aug_inc_logits,batch_y)

                    self.train_loss_incoming.append(der_loss.item())

                    mem_x, mem_y,mem_logits = self.memory_manager.retrieve_from_mem_logits(batch_x, batch_y,
                                                                         retrieve_num=10)
                    aug_batch_x = self.aug_data(batch_x)
                    aug_inc_logits = self.model.forward(aug_batch_x)
                    if mem_x.shape[0]>0:
                        aug_mem_x = self.aug_data(mem_x)
                        aug_mem_logits = self.model.forward(aug_mem_x)
                        mse_loss = torch.nn.MSELoss()
                        mem_reg_loss = self.alpha * mse_loss(aug_mem_logits,mem_logits)
                        #self.train_loss_mem.append(mem_reg_loss.item())
                        der_loss += mem_reg_loss

                    self.loss_batch.append(der_loss.item())
                    losses_batch.update(der_loss.item(), batch_y.size(0))

                    der_loss.backward()
                    self.opt.step()
                end_time = time.time()




                self.mem_iter_list.append(memiter)
                # if(self.params.use_test_buffer):
                #     self.close_loop_cl.compute_testmem_loss()
                self.memory_manager.update_memory_logits(batch_x, batch_y,aug_inc_logits.data)

                if i % 100 == 1 and self.verbose and ep %10 ==0:
                    print(
                        '==>>> it: {}, avg. loss: {:.3f}, '
                        'running train acc: {:.2f} '
                        'ep:{}'
                            .format(i, losses_batch.avg(), acc_batch.avg(),ep)
                    )
                    # print(
                    #     '==>>> it: {}, mem avg. loss: {:.6f}, '
                    #     'running mem acc: {:.3f}'
                    #         .format(i, losses_mem.avg(), acc_mem.avg())
                    # )
                    print('time:{:.2f} '
                          "memloss:{:.2f} "
                    'derloss:{:.2f} '
                          .format(end_time-start_time,mem_reg_loss.item(),der_loss.item()))
                    #print("mem_iter", "time:",end_time-start_time,"memloss:",mem_reg_loss.item(),"derloss",der_loss.item())
        self.after_train()
        return test_acc_list
