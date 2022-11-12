from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
from utils.setup_elements import input_size_match
from torchvision.transforms import transforms
from utils.augmentations import RandAugment
import torch
from utils.utils import maybe_cuda
#import imgaug.augmenters as iaa
import numpy as np

class aug_agent(object):
    def __init__(self,params,CL_agent=None):
        self.params = params
        self.CL_agent = CL_agent
        _CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

        self.transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])

        self.current_N = self.params.randaug_N
        self.current_M = self.params.randaug_M

        self.transform_train_mem = self.transform_train
        self.transform_train_incoming = self.transform_train
        #self.aug = iaa.RandAugment(n=self.params.randaug_N, m=self.params.randaug_M)
        self.transform_train.transforms.insert(0, RandAugment(self.params.randaug_N, self.params.randaug_M))
        self.transform_train_mem.transforms.insert(0, RandAugment(self.params.randaug_N, self.params.randaug_M))
        self.transform_train_incoming.transforms.insert(0, RandAugment(self.params.randaug_N, self.params.randaug_M))

        self.scr_transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]),
                              scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)

        )

        if (self.params.aug_target == "both"):
            self.mem_aug = True
            self.incoming_aug = True
        elif (self.params.aug_target == "mem"):
            self.mem_aug = True
            self.incoming_aug = False
        elif (self.params.aug_target == "incoming"):
            self.mem_aug = False
            self.incoming_aug = True
        else:
            self.mem_aug = False
            self.incoming_aug = False
    def set_deraug(self):
        # if(self.params.aug_normal):
        #     self.transform_train = transforms.Compose([
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5071, 0.4867, 0.4408),
        #                                   (0.2675, 0.2565, 0.2761)),
        #         #transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        #     ])
        # else:
        size = input_size_match[self.params.data][-1]
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        self.transform_train_mem = self.transform_train
        self.transform_train_incoming = self.transform_train
        #assert False

    def set_aug_NM(self, N,M):
        self.transform_train_mem = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])


        self.transform_train_mem.transforms.insert(0, RandAugment(N,M))
        #self.aug = iaa.RandAugment(n=N, m=M)
        self.transform_train_incoming = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])


        self.transform_train_incoming.transforms.insert(0, RandAugment(N,M))



    def set_aug_para(self, N_mem, N_incoming,):
        self.transform_train_mem = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        # ])

        self.transform_train_mem.transforms.insert(0, RandAugment(N_mem, self.current_M))
        #self.aug = iaa.RandAugment(n=N, m=M)

        self.transform_train_incoming = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])

        self.transform_train_incoming.transforms.insert(0, RandAugment(N_incoming, self.current_M))


    def aug_data_old_batch(self,concat_batch_x):
        #print("aug")
        n, c, w, h = concat_batch_x.shape

        all_images = [transforms.ToPILImage()(concat_batch_x[i]) for i in range(n)]
        aug_images = [self.transform_train_mem(image).reshape([1, c, w, h]) for image in all_images]


        aug_concat_batch_x = maybe_cuda(torch.cat((aug_images), dim=0))

        return aug_concat_batch_x
    def aug_data_old(self,concat_batch_x,mem_num):
        #use torchvision.transforms library for augmentation implementation
        # if (self.params.randaug_type == "dynamic"):
        #     self.set_aug_para(1, self.CL_agent.task_seen)
        n, c, w, h = concat_batch_x.shape

        mem_images = [transforms.ToPILImage()(concat_batch_x[i]) for i in range(mem_num)]
        incoming_images = [transforms.ToPILImage()(concat_batch_x[i]) for i in range(mem_num,n)]
        if(self.mem_aug and mem_num>0):
            aug_mem = [self.transform_train_mem(image).reshape([1, c, w, h]) for image in mem_images]
            aug_mem = maybe_cuda(torch.cat(aug_mem,dim=0))
        else:
            aug_mem = concat_batch_x[:mem_num,:,:,:]
        if(self.incoming_aug):
            aug_incoming = [self.transform_train_incoming(image).reshape([1, c, w, h]) for image in incoming_images]
            aug_incoming = maybe_cuda(torch.cat(aug_incoming,dim=0))
        else:
            aug_incoming = concat_batch_x[mem_num:,:,:,:]


        if(mem_num>0):
            aug_concat_batch_x = maybe_cuda(torch.cat((aug_mem,aug_incoming), dim=0))
        else:
            aug_concat_batch_x = maybe_cuda(aug_incoming)

        return aug_concat_batch_x
    # def aug_data(self,concat_batch_x,mem_num):
    ## use iaa.RandAugment library for augmentation implementation
    #     if(self.params.randaug_type == "dynamic"):
    #         self.set_aug_para(1,self.CL_agent.task_seen)
    #     n, c, w, h = concat_batch_x.shape
    #     #
    #     concat_batch_x =concat_batch_x.cpu().numpy().astype(np.uint8)
    #     aug_concat_batch_x = self.aug(images = concat_batch_x.reshape((n,w,h,c)))
    #     aug_concat_batch_x = aug_concat_batch_x.astype(np.float32).reshape((n,c,w,h))
    #     aug_concat_batch_x = maybe_cuda(torch.tensor(aug_concat_batch_x))


    #     return aug_concat_batch_x
    def scr_aug_data(self,concat_batch_x,mem_num):

        n, c, w, h = concat_batch_x.shape

        mem_images = concat_batch_x[:mem_num,:,:,:]
        incoming_images = concat_batch_x[mem_num:,:,:,:]
        if(self.mem_aug and mem_num>0):
            aug_mem =self.scr_transform(mem_images)
        else:
            aug_mem = mem_images #concat_batch_x[:mem_num,:,:,:]
        if(self.incoming_aug):
            aug_incoming = self.scr_transform(incoming_images) #[self.transform_train(image).reshape([1, c, w, h]) for image in incoming_images]
        else:
            aug_incoming = incoming_images #concat_batch_x[mem_num:,:,:,:]


        if(mem_num>0):
            aug_concat_batch_x = maybe_cuda(torch.cat((aug_mem,aug_incoming), dim=0))
        else:
            aug_concat_batch_x = maybe_cuda(aug_incoming)

        return aug_concat_batch_x



