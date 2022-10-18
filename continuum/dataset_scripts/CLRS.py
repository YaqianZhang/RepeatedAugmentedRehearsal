import os
from continuum.dataset_scripts.dataset_base import DatasetBase
import pickle as pkl
import logging
from hashlib import md5
import numpy as np
from PIL import Image
from continuum.data_utils import shuffle_data, load_task_with_labels
import time
import pandas as pd
import random


CLRS25_ntask = {
    'ni': 3,
    'nc': 5,
    'nic':15

}


class CLRS25(DatasetBase):
    def __init__(self, scenario, params):
        if isinstance(params.num_runs, int) and params.num_runs > 4:
            raise Exception('the max number of runs for CLRS25 is 4')
        dataset = 'clrs25'
        task_nums = CLRS25_ntask[scenario]
        if(params.offline):
            task_nums = 1
        self.scenario =scenario

        super(CLRS25, self).__init__(dataset, scenario, task_nums, params.num_runs, params)


    def download_load(self):

        self.root_folder = self.root




    def setup(self, cur_run):
        self.task_labels =[]
        if(self.params.cl_type == "ni"):
            label = "/label/NI/run"
        elif(self.params.cl_type == "nc"):
            label = "/label/NC/run"
        else:
            raise NotImplementedError("cl_type")


        for task_id in range(1,self.params.num_tasks+1):


            train_path_file = self.root + label + str(cur_run) + "/train_task" + str(task_id) + ".txt"
            train_data = pd.read_csv(train_path_file, delimiter="\t")

            train_label = list(train_data.values[:, 1])
            self.task_labels.append(set(train_label))



        if(self.params.offline):
            labels = np.arange(25)
            self.task_labels = [set(labels)]




        test_path_file = self.root +label+str(cur_run)+"/test.txt"
        test_data = pd.read_csv(test_path_file, delimiter="\t")
        test_idx_list = list(test_data.values[:, 0])
        test_label = list(test_data.values[:, 1])
        self.val_set = []
        self.test_set = []
        print('Loading test set...')

        #test paths
        test_paths = []
        for idx in test_idx_list:
            test_paths.append(os.path.join(self.root, "CLRS/"+idx))

        # test imgs
        self.test_data = self.get_batch_from_paths(test_paths)

        self.test_label = np.asarray(test_label)



        # for i,l in enumerate(test_label):
        #     x_test = self.test_data[i]
        #     y_test = self.test_label[i]
        #     self.test_set.append((x_test, y_test))



        if self.scenario == 'nc'  :

            for labels in self.task_labels:
                labels = list(set(labels))
                print(labels)
                x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)

                self.test_set.append((x_test, y_test))
        else:
            labels = list(set(self.task_labels[0]))
            print(labels)
            x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)

            self.test_set.append((x_test, y_test))

        # elif self.scenario == 'ni':
        #     self.test_set = [(self.test_data, self.test_label)]

        if (self.params.offline):
            self.train_id_all = []
            self.train_label_all =[]
            for task_id in range(5):
                train_path_file = self.root + label + str(cur_run) + "/train_task" + str(
                    task_id + 1) + ".txt"
                train_data = pd.read_csv(train_path_file, delimiter="\t")
                train_idx_list = list(train_data.values[:, 0])
                train_label = list(train_data.values[:, 1])
                self.train_id_all += train_idx_list
                self.train_label_all += train_label

    def new_task(self, cur_task, **kwargs):
        cur_run = kwargs['cur_run']
        s = time.time()
        if(self.params.cl_type == "nc"):
            train_path_file = self.root +"/label/NC/run"+str(cur_run)+"/train_task"+str(cur_task+1)+".txt"
        elif(self.params.cl_type == "ni"):

            train_path_file = self.root + "/label/NI/run" + str(cur_run) + "/train_task" + str(cur_task + 1) + ".txt"
        else:
            raise NotImplementedError("cl_type")
        train_data = pd.read_csv(train_path_file, delimiter="\t")
        train_idx_list = list(train_data.values[:, 0])
        train_label = list(train_data.values[:, 1])





        if (self.params.offline):
            # train_len = len(train_idx_list)
            # total_len = len(self.train_id_all)
            #
            # idx = np.random.randint(0,total_len,train_len)
            # train_idx_list = [self.train_id_all[i] for i in idx]
            # train_label = [self.train_label_all[i] for i in idx]
            indices = np.arange(len(self.train_label_all))
            np.random.shuffle(indices)

            #random.shuffle(self.train_id_all)
            train_idx_list =list( np.array(self.train_id_all)[indices])
            #random.shuffle(self.train_label_all)
            train_label = list(np.array(self.train_label_all)[indices])






            #train_idx_list = self.LUP[self.scenario][cur_run][cur_task]
        print("Loading data...")
        # Getting the actual paths
        train_paths = []
        for idx in train_idx_list:
            train_paths.append(os.path.join(self.root,"CLRS/"+ idx))
        # loading imgs
        train_x = self.get_batch_from_paths(train_paths)

        train_y = train_label
        train_y = np.asarray(train_y)

        # get val set
        train_x_rdm, train_y_rdm = shuffle_data(train_x, train_y)
        val_size = int(len(train_x_rdm) * self.params.val_size)
        val_data_rdm, val_label_rdm = train_x_rdm[:val_size], train_y_rdm[:val_size]
        train_data_rdm, train_label_rdm = train_x_rdm[val_size:], train_y_rdm[val_size:]
        self.val_set.append((val_data_rdm, val_label_rdm))
        e = time.time()
        print('loading time {}'.format(str(e-s)))
        return train_data_rdm, train_label_rdm, set(train_label_rdm)


    def new_run(self, **kwargs):
        cur_run = kwargs['cur_run']
        self.setup(cur_run)


   # @staticmethod
    def get_batch_from_paths(self,paths, compress=False, snap_dir='',
                             on_the_fly=True, verbose=False):
        """ Given a number of abs. paths it returns the numpy array
        of all the images. """

        # Getting root logger
        log = logging.getLogger('mylogger')

        # If we do not process data on the fly we check if the same train
        # filelist has been already processed and saved. If so, we load it
        # directly. In either case we end up returning x and y, as the full
        # training set and respective labels.
        num_imgs = len(paths)
        hexdigest = md5(''.join(paths).encode('utf-8')).hexdigest()
        log.debug("Paths Hex: " + str(hexdigest))
        loaded = False
        x = None
        file_path = None

        if compress:
            file_path = snap_dir + hexdigest + ".npz"
            if os.path.exists(file_path) and not on_the_fly:
                loaded = True
                with open(file_path, 'rb') as f:
                    npzfile = np.load(f)
                    x, y = npzfile['x']

        else:
            x_file_path = snap_dir + hexdigest + "_x.bin"
            if os.path.exists(x_file_path) and not on_the_fly:
                loaded = True
                with open(x_file_path, 'rb') as f:

                    x = np.fromfile(f, dtype=np.uint8) \
                        .reshape(num_imgs, 256,256, 3)

        # Here we actually load the images.
        if not loaded:
            # Pre-allocate numpy arrays
            x = np.zeros((num_imgs, 128,128, 3), dtype=np.uint8)

            for i, path in enumerate(paths):
                if verbose:
                    print("\r" + path + " processed: " + str(i + 1), end='')
                load_img = np.array(Image.open(path))
                x[i]= np.resize(load_img,(128,128,3))


            if verbose:
                print()

            if not on_the_fly:
                # Then we save x
                if compress:
                    with open(file_path, 'wb') as g:
                        np.savez_compressed(g, x=x)
                else:
                    x.tofile(snap_dir + hexdigest + "_x.bin")

        assert (x is not None), 'Problems loading data. x is None!'

        return x
