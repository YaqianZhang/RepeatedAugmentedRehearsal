import time
import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
# from RL.pytorch_util import  build_mlp
import numpy as np
from utils.utils import cutmix_data
from torchvision.transforms import transforms
from utils.augmentations import RandAugment

class SupContrastReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(SupContrastReplay, self).__init__(model, opt, params)
        #self.buffer = Buffer(model, params)
        self.buffer = self.memory_manager.buffer
        self.close_loop_cl = None
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)

        )
        if(self.params.aug_type == "two"):
            self.transform = nn.Sequential(
                RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]),
                                  scale=(0.2, 1.)),
                ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8)


            )
        if(params.data in [ 'clrs25', 'core50']):
            softmax_inputdim = 2560
        elif(params.data in ['cifar100','cifar10',]):
            softmax_inputdim = 160

        elif (params.data in [ "mini_imagenet"]):
            softmax_inputdim = 640
        else:
            raise NotImplementedError("undefined dataset",params.data)
        print(softmax_inputdim)

        # if(params.softmax_type == "seperate"):
        #     self.init_seperate_softmax(softmax_inputdim)
        # elif(params.softmax_type == "meta"):
        #     self.meta_softmax(softmax_inputdim)



        # self.replay_para={"mem_ratio":self.params.mem_ratio,
        #                   "incoming_ratio":self.params.incoming_ratio,
        #                   "mem_iter":self.params.mem_iters}

        self.replay_para = {"mem_ratio": self.params.mem_ratio,
                            "incoming_ratio": self.params.incoming_ratio,
                            "mem_iter": self.params.mem_iters,
                            "randaug_M": self.params.randaug_M}


    # def init_seperate_softmax(self,softmax_inputdim):
    #     self.softmax_head = maybe_cuda(build_mlp(input_size=softmax_inputdim,
    #         output_size=100,
    #         n_layers=self.params.softmax_nlayers,
    #         size=self.params.softmax_nsize,
    #         use_dropout=self.params.softmax_dropout,))
    #     self.softmax_opt =  torch.optim.SGD(self.softmax_head .parameters(),
    #                             lr=self.params.softmaxhead_lr,
    #                             )





   # ## override base function
   #  def compute_nmc_mean(self):
   #      exemplar_means = {}
   #      cls_exemplar = {cls: [] for cls in self.old_labels}
   #      # buffer_filled = self.buffer.current_index
   #      # for x, y in zip(self.buffer.buffer_img[:buffer_filled],
   #      #                 self.buffer.buffer_label[:buffer_filled]):
   #      buffer_filled = self.memory_manager.buffer.current_index
   #      for x, y in zip(self.memory_manager.buffer.buffer_img[:buffer_filled],
   #                      self.memory_manager.buffer.buffer_label[:buffer_filled]):
   #          x = maybe_cuda(x)
   #          y = maybe_cuda(y)
   #          cls_exemplar[y.item()].append(x)
   #      for cls, exemplar in cls_exemplar.items():
   #          features = []
   #          # Extract feature for each exemplar in p_y
   #          for ex in exemplar:
   #              feature = self.model.features(ex.unsqueeze(0)).detach().clone()
   #              feature = feature.squeeze()
   #              feature.data = feature.data / feature.data.norm()  # Normalize
   #              features.append(feature)
   #          if len(features) == 0:
   #              mu_y = maybe_cuda(
   #                  torch.normal(0, 1, size=tuple(self.model.features(x.unsqueeze(0)).detach().size())), self.cuda)
   #              mu_y = mu_y.squeeze()
   #          else:
   #              features = torch.stack(features)
   #              mu_y = features.mean(0).squeeze()
   #          mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
   #          exemplar_means[cls] = mu_y
   #      return exemplar_means




    def _compute_softmax_logits(self,x,need_grad = True):
        if(need_grad == False):
            with torch.no_grad():
                h_feature = self.model.features(x)
                logits = self.softmax_head(h_feature)
        else:
            h_feature = self.model.features(x)
            logits = self.softmax_head(h_feature)
        return logits


    def perform_scr_update(self,combined_batch, combined_labels,):


        ######## scr loss
        if (self.params.no_aug):
            combined_batch_aug = combined_batch.clone()
        else:

            combined_batch_aug = self.transform(combined_batch)

        features = torch.cat([self.model.forward(combined_batch).unsqueeze(1),
                              self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
        loss, loss_full = self.criterion(features, combined_labels)



        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.loss_batch.append(loss.item())


        #print(loss)
        #softmax_loss,acc_incoming,incoming_loss = self.perform_softmax_training(combined_batch, combined_labels, mem_x.shape[0])

        #return loss,softmax_loss,acc_incoming,incoming_loss

        return loss.item(),



    def perform_softmax_update(self,x,y,mem_num,replay_para = None):
        if(self.params.softmax_type == "None"):

            return
        if(replay_para == None):
            replay_para = self.replay_para


        total_num = x.shape[0]

        logits = self._compute_softmax_logits(x)

        ce_all = torch.nn.CrossEntropyLoss(reduction='none')
        softmax_loss_full = ce_all(logits, y)

        mem_loss = torch.mean(softmax_loss_full[:mem_num])
        incoming_loss = torch.mean(softmax_loss_full[mem_num:])

        _, pred_label = torch.max(logits, 1)
        acc = (pred_label == y)
        if(mem_num >0):
            acc_mem =  acc[:mem_num].sum().item() /mem_num
        else:
            acc_mem = 0
        acc_incoming = acc[mem_num:].sum().item() / (total_num -mem_num)

        self.train_loss_mem.append(mem_loss.item())
        self.train_loss_incoming.append(incoming_loss.item())
        self.train_acc_mem.append(acc_mem)
        self.train_acc_incoming.append(acc_incoming)


        # if (self.close_loop_cl != None):
        #     self.close_loop_cl.last_train_loss = mem_loss.item()

        softmax_loss = (replay_para['mem_ratio']*torch.sum(softmax_loss_full[:mem_num]) +\
                replay_para['incoming_ratio'] * torch.sum(softmax_loss_full[mem_num:]))/total_num


        self.softmax_opt.zero_grad()
        softmax_loss.backward()
        self.softmax_opt.step()
        #print("softmax",self.params.incoming_ratio,mem_num,len(y))
        #self.perform_cutmix( x, y)
        return acc_mem



    def train_learner(self, x_train, y_train):


        #self.memory_manager.reset_new_old()
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses = AverageMeter()
        acc_batch = AverageMeter()


        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                #batch_x, batch_y = self.memory_manager.update_before_training(batch_x, batch_y)

                self.set_memIter()
                for j in range(self.mem_iters):
                    #s = time.time()
                    concat_batch_x, concat_batch_y, mem_num = self.concat_memory_batch(batch_x, batch_y)
                   # e1 = time.time()
                    scr_loss = self.perform_scr_update(concat_batch_x, concat_batch_y)
                    #losses.update(scr_loss,batch_y.size(0))
                    acc_mem = self.perform_softmax_update(concat_batch_x, concat_batch_y,mem_num)
                    # e2 = time.time()
                    # print("aug",e1-s,"scr update",e2-e1)


                # update mem
                self.memory_manager.update_memory(batch_x, batch_y)
                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                            .format(i, losses.avg(), acc_batch.avg())
                    )
                    print("Iter", self.mem_iters,concat_batch_y.shape)


        self.after_train()



