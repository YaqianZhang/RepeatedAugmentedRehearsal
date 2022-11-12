import time
from abc import abstractmethod
import abc
import numpy as np
import torch
from torch.nn import functional as F
from utils.kd_manager import KdManager
from utils.utils import maybe_cuda, AverageMeter
from torch.utils.data import TensorDataset, DataLoader
import copy
from utils.loss import SupConLoss
import pickle
from utils.buffer.memory_manager import memory_manager_class
from utils.buffer.memory_manager_2buff import memory_manager_class_2buff
from utils.utils import cutmix_data
from utils.buffer.buffer_utils import get_grad_vector
import copy
from utils.aug_agent import aug_agent

from sklearn.metrics import confusion_matrix, cohen_kappa_score


class ContinualLearner(torch.nn.Module, metaclass=abc.ABCMeta):
    '''
    Abstract module which is inherited by each and every continual learning algorithm.
    '''

    def __init__(self, model, opt, params):
        super(ContinualLearner, self).__init__()
        self.params = params
        self.model = model
        self.opt = opt
        self.data = params.data
        self.cuda = params.cuda
        self.epoch = params.epoch
        self.batch = params.batch
        self.verbose = params.verbose
        self.initialize()
        self.RL_replay = None

        if(self.params.switch_buffer_type=="two_buffer"):
            self.memory_manager = memory_manager_class_2buff(model,params)
        else:

            self.memory_manager = memory_manager_class(model, params)
        self.aug_agent = aug_agent(params,CL_agent=self)

        ## save train acc
        self.mem_iter_list =[]
        self.mem_batchsize_list=[]
        self.aug_N_list=[]
        self.incoming_ratio_list=[]
        self.mem_ratio_list=[]

        self.train_acc_incoming = []
        self.train_acc_mem=[]
        self.train_acc_mem_main=[]
        self.train_acc_mem_add=[]
        self.test_acc_mem=[]
        self.test_acc_true=[]
        self.test_loss_true=[]

        self.train_acc_incoming_org = []
        self.train_acc_mem_org=[]
        self.train_loss_incoming_org = []
        self.train_loss_mem_org=[]

        self.restart_flag =[]

        self.train_loss_incoming = []
        self.train_loss_mem=[]
        self.test_loss_mem=[]
        self.test_loss_mem_new=[]
        self.test_loss_mem_old=[]
        self.train_loss_old=[]
        self.loss_batch=[]
        self.train_acc_blc=[]
        self.train_loss_blc=[]

        self.adaptive_learning_rate=[]
    def initialize(self):
        print("base agent initialize")
        self.old_labels = []
        self.new_labels = []
        self.task_seen = 0
        self.kd_manager = KdManager()
        self.error_list = []
        self.new_class_score = []
        self.old_class_score = []
        self.fc_norm_new = []
        self.fc_norm_old = []
        self.bias_norm_new = []
        self.bias_norm_old = []
        self.lbl_inv_map = {}
        self.class_task_map = {}

    def before_train(self, x_train, y_train):
        new_labels = list(set(y_train.tolist()))
        self.new_labels += new_labels
        for i, lbl in enumerate(new_labels):
            self.lbl_inv_map[lbl] = len(self.old_labels) + i
        #print("set new labels",self.lbl_inv_map)

        for i in new_labels:
            self.class_task_map[i] = self.task_seen

    @abstractmethod
    def train_learner(self, x_train, y_train):
        pass

    def after_train(self):
        #self.old_labels = list(set(self.old_labels + self.new_labels))
        self.old_labels += self.new_labels
        self.new_labels_zombie = copy.deepcopy(self.new_labels)
        self.new_labels.clear()
        self.task_seen += 1
        if self.params.trick['review_trick'] and hasattr(self, 'buffer'):
            self.model.train()
            mem_x = self.buffer.buffer_img[:self.buffer.current_index]
            mem_y = self.buffer.buffer_label[:self.buffer.current_index]
            # mem_x = maybe_cuda(mem_x)
            # mem_y = maybe_cuda(mem_y)
            # criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            if mem_x.size(0) > 0:
                rv_dataset = TensorDataset(mem_x, mem_y)
                rv_loader = DataLoader(rv_dataset, batch_size=self.params.eps_mem_batch, shuffle=True, num_workers=0,
                                       drop_last=True)
                for ep in range(1):
                    for i, batch_data in enumerate(rv_loader):
                        # batch update
                        batch_x, batch_y = batch_data
                        batch_x = maybe_cuda(batch_x, self.cuda)
                        batch_y = maybe_cuda(batch_y, self.cuda)
                        logits = self.model.forward(batch_x)
                        if self.params.agent[:3] == 'SCR':
                            logits = torch.cat([self.model.forward(batch_x).unsqueeze(1),
                                                self.model.forward(self.transform(batch_x)).unsqueeze(1)], dim=1)
                        loss = self.criterion(logits, batch_y)
                        self.opt.zero_grad()
                        loss.backward()
                        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                        grad = [p.grad.clone() / 10. for p in params]
                        for g, p in zip(grad, params):
                            p.grad.data.copy_(g)
                        self.opt.step()

        if self.params.trick['kd_trick'] or self.params.agent == 'LWF':
            self.kd_manager.update_teacher(self.model)

    def criterion(self, logits, labels,reduction_type ="mean"):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction=reduction_type)

        if self.params.trick['labels_trick']:
            unq_lbls = labels.unique().sort()[0]
            for lbl_idx, lbl in enumerate(unq_lbls):
                labels[labels == lbl] = lbl_idx
            # Calcualte loss only over the heads appear in the batch:
            return ce(logits[:, unq_lbls], labels)
        elif self.params.trick['separated_softmax']:
            old_ss = F.log_softmax(logits[:, self.old_labels], dim=1)
            new_ss = F.log_softmax(logits[:, self.new_labels], dim=1)
            ss = torch.cat([old_ss, new_ss], dim=1)
            for i, lbl in enumerate(labels):
                labels[i] = self.lbl_inv_map[lbl.item()]
            return F.nll_loss(ss, labels)
        elif self.params.agent in ['SCR','SCR_META', 'SCP',"SCR_RL_ratio","SCR_RL_iter"]:
            SC = SupConLoss(temperature=self.params.temp)
            return SC(logits, labels)
        else:
            return ce(logits, labels)
    def adaptive_criterion(self,logits,labels,ratio):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='none')
        print(ce,ratio)
        assert False
        return torch.mean(ce(logits,labels)*ratio)

    def forward(self, x):
        return self.model.forward(x)
    def forward_with_weight(self, x,weight):
        return self.model.forward_with_weight(x,weight)



    def set_memIter(self):
        #print(self.task_seen,self.params.aug_start)
        if self.task_seen >= self.params.aug_start:
            self.mem_iters = self.params.mem_iters

            #### todo: mir it is subsample

            self.memory_manager.buffer.retrieve_method.num_retrieve = self.params.eps_mem_batch

        else:
            self.mem_iters = 1
            #print(self.params.agent)
            if(self.params.agent == "SCR"):
                pass

                #self.memory_manager.buffer.retrieve_method.num_retrieve = self.params.eps_mem_batch
            else:
                self.memory_manager.buffer.retrieve_method.num_retrieve = 10


    def concat_memory_batch(self, batch_x,batch_y,retrieve_num=None):
        mem_x, mem_y = self.memory_manager.retrieve_from_mem(batch_x, batch_y, self.task_seen,retrieve_num=retrieve_num)
        if mem_x.size(0) > 0:
            mem_x = maybe_cuda(mem_x, self.cuda)
            mem_y = maybe_cuda(mem_y, self.cuda)

            batch_x = torch.cat([mem_x,batch_x,])
            batch_y = torch.cat([mem_y,batch_y,])
        return self.perform_augmentation(batch_x,batch_y,mem_x.size(0))
    def perform_augmentation(self,batch_x,batch_y,mem_batch_size):

        if(self.task_seen >= self.params.aug_start):
            if (self.params.randaug):


                batch_x_aug2 = self.aug_agent.aug_data_old(batch_x, mem_batch_size, )

                batch_x = batch_x_aug2

            if (self.params.scraug):
                batch_x = self.aug_agent.scr_aug_data(batch_x,mem_batch_size)

        return batch_x,batch_y,mem_batch_size
    def _compute_softmax_logits(self,x,need_grad = True):

        if(need_grad):

            logits = self.model.forward(x)
        else:
            with torch.no_grad():
                logits = self.model.forward(x)
        return logits

    def compute_acc(self,logits,batch_y):
        # if (self.params.only_task_seen):
        #     labels_seen = self.old_labels + self.new_labels
        #     _, pred_label_idx = torch.max(logits[:, labels_seen], 1)
        #     pred_label= [labels_seen[idx] for idx in pred_label_idx]
        #     correct = [pred_label[i] == batch_y[i].item() for i in range(len(pred_label))]
        #     #print(correct,pred_label_true,batch_y)
        #
        #
        #     correct_cnt = np.mean(correct)
        #     # print("task seen", pred_label_true,  labels_seen,correct_cnt)
        #     #
        #     # _, pred_label = torch.max(logits, 1)
        #     #
        #     # correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
        #     # print( "task all",pred_label, batch_y,correct_cnt)
        #     # assert False
        # else:
        _, pred_label = torch.max(logits, 1)
        correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)


        # for x in batch_y:
        #
        #     if (x.item() not in self.lbl_inv_map.keys()):        #         print("unseed", x.item())
        return correct_cnt,pred_label

    def compute_nmc_mean(self):
        exemplar_means = {}
        cls_exemplar = {cls: [] for cls in self.old_labels}
        buffer_filled = self.buffer.current_index
        for x, y in zip(self.buffer.buffer_img[:buffer_filled],
                        self.buffer.buffer_label[:buffer_filled]):
        # buffer_filled = self.memory_manager.buffer.current_index
        # for x, y in zip(self.memory_manager.buffer.buffer_img[:buffer_filled],
        #                 self.memory_manager.buffer.buffer_label[:buffer_filled]):
            x = maybe_cuda(x)
            y = maybe_cuda(y)
            cls_exemplar[y.item()].append(x)
        for cls, exemplar in cls_exemplar.items():
            features = []
            # Extract feature for each exemplar in p_y
            for ex in exemplar:
                feature = self.model.features(ex.unsqueeze(0)).detach().clone()
                feature = feature.squeeze()
                feature.data = feature.data / feature.data.norm()  # Normalize
                features.append(feature)
            if len(features) == 0:
                mu_y = maybe_cuda(
                    torch.normal(0, 1, size=tuple(self.model.features(x.unsqueeze(0)).detach().size())), self.cuda)
                mu_y = mu_y.squeeze()
            else:
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
            mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
            exemplar_means[cls] = mu_y
        return exemplar_means
    def nmc_predict(self,batch_x,batch_y,exemplar_means):
        feature = self.model.features(batch_x)  # (batch_size, feature_size)
        for j in range(feature.size(0)):  # Normalize
            feature.data[j] = feature.data[j] / feature.data[j].norm()
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        means = torch.stack(
            [exemplar_means[cls] for cls in self.old_labels])  # (n_classes, feature_size)

        # old ncm
        means = torch.stack([means] * batch_x.size(0))  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)
        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)
        dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
        min_dist, preds = dists.min(1)
        correct_cnt = (np.array(self.old_labels)[
                           preds.tolist()] == batch_y.cpu().numpy()).sum().item() / batch_y.size(0)

        ## todo:zyq how to compute loss for icarl
        # logits = self.model.forward(batch_x)
        # _, pred_label = torch.max(logits, 1)
        # loss_batch = self.criterion(logits, batch_y)

        loss_batch = min_dist.mean().item()
        return correct_cnt,preds,loss_batch

    def softmax_evaluate(self, test_loaders, ):
        self.model.eval()
        acc_array = np.zeros(len(test_loaders))
        loss_array = np.zeros(len(test_loaders))
        #kappa_array = np.zeros(len(test_loaders))

        with torch.no_grad():

            for task, test_loader in enumerate(test_loaders):
                acc = AverageMeter()
                loss = AverageMeter()
                kappa_score = AverageMeter()
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)


                    if (task >= self.task_seen):
                        break
                    if (self.params.agent[:3] == "SCR" and self.params.use_softmaxloss):
                        features = self.model.features(batch_x)
                        logits = self.softmax_head(features)
                        correct_cnt, pred_label = self.compute_acc(logits, batch_y)
                        loss_batch = torch.nn.CrossEntropyLoss()(logits, batch_y)
                    else:

                        logits = self.model.forward(batch_x)
                        correct_cnt, pred_label = self.compute_acc(logits, batch_y)
                        loss_batch = self.criterion(logits, batch_y)
                    # kappa = cohen_kappa_score(pred_label.cpu().numpy(), batch_y.cpu().numpy())
                    # kappa_score.update(kappa, batch_y.size(0))
                    acc.update(correct_cnt, batch_y.size(0))
                    loss.update(loss_batch, batch_y.size(0))
                acc_array[task] = acc.avg()
                loss_array[task] = loss.avg()

        return acc_array, loss_array


    def evaluate(self, test_loaders,no_nmc=False):
        self.model.eval()
        acc_array = np.zeros(len(test_loaders))
        loss_array = np.zeros(len(test_loaders))
        kappa_array=np.zeros(len(test_loaders))
        if self.params.trick['nmc_trick'] or self.params.agent in ['ICARL','SCP']:
            if(no_nmc):
                pass
            else:

                exemplar_means =self.compute_nmc_mean()


        with torch.no_grad():
            if self.params.error_analysis:
                error = 0
                no = 0
                nn = 0
                oo = 0
                on = 0
                new_class_score = AverageMeter()
                old_class_score = AverageMeter()
                correct_lb = []
                predict_lb = []
            for task, test_loader in enumerate(test_loaders):
                acc = AverageMeter()
                loss = AverageMeter()
                kappa_score = AverageMeter()
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)
                    if no_nmc == False and (self.params.trick['nmc_trick'] or self.params.agent in ['ICARL', 'SCP']):

                        correct_cnt,pred_label,loss_batch = self.nmc_predict(batch_x,batch_y,exemplar_means)
                        # feature = self.model.features(batch_x)  # (batch_size, feature_size)
                        # for j in range(feature.size(0)):  # Normalize
                        #     feature.data[j] = feature.data[j] / feature.data[j].norm()
                        # feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
                        # means = torch.stack(
                        #     [exemplar_means[cls] for cls in self.old_labels])  # (n_classes, feature_size)
                        #
                        # # old ncm
                        # means = torch.stack([means] * batch_x.size(0))  # (batch_size, n_classes, feature_size)
                        # means = means.transpose(1, 2)
                        # feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)
                        # dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
                        # _, preds = dists.min(1)
                        # correct_cnt = (np.array(self.old_labels)[
                        #                    preds.tolist()] == batch_y.cpu().numpy()).sum().item() / batch_y.size(0)
                        #
                        # ## todo:zyq how to compute loss for icarl
                        # logits = self.model.forward(batch_x)
                        # _, pred_label = torch.max(logits, 1)
                        # #loss_batch = self.criterion(logits, batch_y)
                        # loss_batch = 0

                    else:

                        if(task>=self.task_seen):
                            break
                        if(self.params.agent[:3] == "SCR" and self.params.use_softmaxloss):
                            features = self.model.features(batch_x)
                            logits = self.softmax_head(features)
                            correct_cnt,pred_label = self.compute_acc(logits,batch_y)
                            loss_batch = torch.nn.CrossEntropyLoss()(logits, batch_y)
                        else:

                            logits = self.model.forward(batch_x)
                            correct_cnt,pred_label = self.compute_acc(logits,batch_y)
                            loss_batch = self.criterion(logits, batch_y)


                        # for x in batch_y:
                        #
                        #     if(x.item() not in self.lbl_inv_map.keys()):
                        #         print("unseed",x.item())
                        #



                    if self.params.error_analysis:
                        correct_lb += [task] * len(batch_y)
                        for i in pred_label:
                            predict_lb.append(self.class_task_map[i.item()])
                        if task < self.task_seen - 1:
                            # old test
                            total = (pred_label != batch_y).sum().item()
                            wrong = pred_label[pred_label != batch_y]
                            error += total
                            on_tmp = sum([(wrong == i).sum().item() for i in self.new_labels_zombie])
                            oo += total - on_tmp
                            on += on_tmp
                            old_class_score.update(
                                logits[:, list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item(),
                                batch_y.size(0))
                        elif task == self.task_seen - 1:
                            # new test
                            total = (pred_label != batch_y).sum().item()
                            error += total
                            wrong = pred_label[pred_label != batch_y]
                            no_tmp = sum([(wrong == i).sum().item() for i in
                                          list(set(self.old_labels) - set(self.new_labels_zombie))])
                            no += no_tmp
                            nn += total - no_tmp
                            new_class_score.update(logits[:, self.new_labels_zombie].mean().item(), batch_y.size(0))
                        else:
                            pass
                    #print(pred_label,batch_y)
                    kappa = cohen_kappa_score(pred_label.cpu().numpy(), batch_y.cpu().numpy())
                    kappa_score.update(kappa,batch_y.size(0))
                    acc.update(correct_cnt, batch_y.size(0))
                    loss.update(loss_batch,batch_y.size(0))
                acc_array[task] = acc.avg()
                loss_array[task] = loss.avg()
                kappa_array[task] = kappa_score.avg()
        print(acc_array)
        #print(kappa_array)

        #print(loss_array)

       # print(task, self.task_seen, "acc", np.mean(acc_array[:self.task_seen]), np.mean(loss_array[:self.task_seen]))
        if self.params.error_analysis:
            self.error_list.append((no, nn, oo, on))
            self.new_class_score.append(new_class_score.avg())
            self.old_class_score.append(old_class_score.avg())
            print("no ratio: {}\non ratio: {}".format(no/(no+nn+0.1), on/(oo+on+0.1)))
            print(self.error_list)
            print(self.new_class_score)
            print(self.old_class_score)
            self.fc_norm_new.append(self.model.linear.weight[self.new_labels_zombie].mean().item())
            self.fc_norm_old.append(self.model.linear.weight[list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item())
            self.bias_norm_new.append(self.model.linear.bias[self.new_labels_zombie].mean().item())
            self.bias_norm_old.append(self.model.linear.bias[list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item())
            print(self.fc_norm_old)
            print(self.fc_norm_new)
            print(self.bias_norm_old)
            print(self.bias_norm_new)
        return acc_array,loss_array

    def save_mem_iters(self,prefix):
        arr = np.array(self.mem_iter_list)
        np.save(prefix + "mem_iter_list", arr)
        arr = np.array(self.mem_batchsize_list)
        np.save(prefix + "mem_batchsize_list", arr)
        arr = np.array(self.aug_N_list)
        np.save(prefix + "aug_N_list",arr)
        arr = np.array(self.incoming_ratio_list)
        np.save(prefix + "incoming_ratio_list", arr)
        arr = np.array(self.mem_ratio_list)
        np.save(prefix + "mem_ratio_list", arr)

    def save_training_acc(self,prefix):
        arr = np.array(self.train_acc_incoming)
        np.save(prefix + "train_acc_incoming.npy", arr)
        arr = np.array(self.restart_flag)
        np.save(prefix +"restart_flag.npy",arr)

        arr = np.array(self.train_acc_mem)
        np.save(prefix + "train_acc_mem.npy", arr)
        arr = np.array(self.train_acc_mem_add)
        np.save(prefix + "train_acc_mem_add.npy", arr)
        arr = np.array(self.train_acc_mem_main)
        np.save(prefix + "train_acc_mem_main.npy", arr)
        arr = np.array(self.mem_batchsize_list)
        np.save(prefix + "mem_batchsize_list", arr)
        arr = np.array(self.mem_iter_list)
        np.save(prefix + "mem_iter.npy", arr)


        arr = np.array(self.aug_N_list)
        np.save(prefix + "aug_N_list",arr)
        arr = np.array(self.train_acc_incoming_org)
        np.save(prefix + "train_acc_incoming_org.npy", arr)

        arr = np.array(self.train_acc_mem_org)
        np.save(prefix + "train_acc_mem_org.npy", arr)

        arr = np.array(self.train_acc_blc)
        np.save(prefix + "train_acc_blc.npy",arr)

        arr = np.array(self.train_loss_blc)
        np.save(prefix +"train_loss_blc.npy",arr)

        arr = np.array(self.adaptive_learning_rate)
        np.save(prefix +"adpt_lr.npy",arr)


        arr = np.array(self.test_acc_true)
        np.save(prefix + "test_acc_true.npy", arr)
        arr = np.array(self.test_loss_true)
        np.save(prefix + "test_loss_true.npy", arr)

        arr = np.array(self.test_acc_mem)
        np.save(prefix + "test_acc_mem.npy", arr)
        arr = np.array(self.train_loss_incoming_org)
        np.save(prefix + "train_loss_incoming_org.npy", arr)

        arr = np.array(self.train_loss_mem_org)
        np.save(prefix + "train_loss_mem_org.npy", arr)

        arr = np.array(self.train_loss_incoming)
        np.save(prefix + "train_loss_incoming.npy", arr)

        arr = np.array(self.train_loss_mem)
        np.save(prefix + "train_loss_mem.npy", arr)
        arr = np.array(self.loss_batch)
        np.save(prefix + "loss_batch.npy", arr)
        arr = np.array(self.test_loss_mem)
        np.save(prefix + "test_loss_mem.npy", arr)
        arr = np.array(self.test_loss_mem_new)
        np.save(prefix + "test_loss_mem_new.npy", arr)
        arr = np.array(self.test_loss_mem_old)
        np.save(prefix + "test_loss_mem_old.npy", arr)

        arr = np.array(self.train_loss_old)
        np.save(prefix + "train_loss_old.npy", arr)
        # if(self.params.RL_type != "NoRL"):
        #
        #     arr = np.array(self.RL_trainer.return_list)
        #     np.save(prefix + "return_list.npy", arr)
        # if(self.params.temperature_scaling):
        #     arr = np.array(self.scaled_model.pre_logits_list)
        #     np.save(prefix + "pre_celoss.npy",arr)
        #     arr = np.array(self.scaled_model.aft_logits_list)
        #     np.save(prefix + "aft_celoss.npy",arr)
        #     arr = np.array(self.scaled_model.opt_temp_list)
        #     np.save(prefix + "opt_temp.npy",arr)



