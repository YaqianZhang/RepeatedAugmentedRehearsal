# Repeated Augmented Rehearsal (RAR) for online continual learning

This is the code repository for Repeated Augmented Rehearsal.

##

## Requirements
![](https://img.shields.io/badge/python-3.7-green.svg)

![](https://img.shields.io/badge/torch-1.5.1-blue.svg)
![](https://img.shields.io/badge/torchvision-0.6.1-blue.svg)
![](https://img.shields.io/badge/PyYAML-5.3.1-blue.svg)
![](https://img.shields.io/badge/scikit--learn-0.23.0-blue.svg)
----
Create a virtual enviroment
```sh
virtualenv rar
```
Activating a virtual environment
```sh
source rar/bin/activate
```
Installing packages
```sh
pip install -r requirements.txt
```

## Run commands

### bash runs
A test run can be performed with the following command:
```
bash run_commands/runs/run_test_rar_er_cifar100.sh
```
Other experiment commands can be found in the folder of [run_commands/runs](run_commands/runs).

Detailed descriptions of options can be found in [general_main.py](general_main.py) and [utils/argparser](utils/argparser)

## Evaluation the results
The results of algorithm outputs will be stored in the folder of [results](results/).

The jupyter notebook [visualize_results.ipynb](visualize_results.ipynb) is used to visualize and analyze results.


## Algorithms 

### Repeated Augmented Rehearsal


### Baselines

* ASER: Adversarial Shapley Value Experience Replay(**AAAI, 2021**) [[Paper]](https://arxiv.org/abs/2009.00093)
* EWC++: Efficient and online version of Elastic Weight Consolidation(EWC) (**ECCV, 2018**) [[Paper]](http://arxiv-export-lb.library.cornell.edu/abs/1801.10112)
* LwF: Learning without forgetting (**ECCV, 2016**) [[Paper]](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_37)
* AGEM: Averaged Gradient Episodic Memory (**ICLR, 2019**) [[Paper]](https://openreview.net/forum?id=Hkf2_sC5FX)
* ER: Experience Replay (**ICML Workshop, 2019**) [[Paper]](https://arxiv.org/abs/1902.10486)
* MIR: Maximally Interfered Retrieval (**NeurIPS, 2019**) [[Paper]](https://proceedings.neurips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html)
* SCR: Supervised Contrastive Replay (**CVPR Workshop, 2021**) [[Paper]](https://arxiv.org/abs/2103.13885) 


## Datasets

### Online Class Incremental

- Split CIFAR100
- Split Mini-ImageNet
- CORe50-NC
- CLRS (Continual Learning Benchmark for Remote
  Sensing Image Scene Classification)
### Data preparation
- CIFAR100 will be downloaded during the first run
- CORE50 download: `source fetch_data_setup.sh`
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download , and place it in datasets/mini_imagenet/
- CLRS: Download from https://github.com/lehaifeng/CLRS



## Acknowledgments
- [SCR/ASER](https://github.com/RaptorMai/online-continual-learning)
- [MIR](https://github.com/optimass/Maximally_Interfered_Retrieval)
- [AGEM](https://github.com/facebookresearch/agem)
