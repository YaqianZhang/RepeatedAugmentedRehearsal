from agents.gdumb import Gdumb
from continuum.dataset_scripts.cifar100 import CIFAR100
from continuum.dataset_scripts.cifar10 import CIFAR10
from continuum.dataset_scripts.core50 import CORE50
from continuum.dataset_scripts.CLRS import CLRS25
from continuum.dataset_scripts.mini_imagenet import Mini_ImageNet
from continuum.dataset_scripts.openloris import OpenLORIS
from agents.exp_replay import ExperienceReplay
from agents.unused.exp_replay_dyna_aug import ExperienceReplay_aug
from agents.unused.exp_replay_dyna_ratio import ExperienceReplay_ratio
from agents.unused.exp_replay_offline import ExperienceReplay_offline
from agents.unused.exp_replay_cl import ExperienceReplay_cl
from agents.unused.exp_replay_cl_bandits import ExperienceReplay_cl_bandits
from agents.unused.exp_replay_cl_bandits_ts import ExperienceReplay_cl_bandits_ts
from agents.unused.exp_replay_batchsize import ExperienceReplay_batchsize
#from unused.rl_exp_replay import RL_ExperienceReplay
from agents.agem import AGEM
from agents.ewc_pp import EWC_pp
from agents.cndpm import Cndpm
from agents.lwf import Lwf
from agents.icarl import Icarl
#from agents.lamaml import LAMAML
from utils.buffer.random_retrieve import Random_retrieve
from utils.buffer.reservoir_update import Reservoir_update

from utils.buffer.mir_retrieve import MIR_retrieve
from utils.buffer.rl_retrieve import RL_retrieve
from utils.buffer.gss_greedy_update import GSSGreedyUpdate
from utils.buffer.aser_retrieve import ASER_retrieve
from utils.buffer.aser_update import ASER_update

from agents.scr import SupContrastReplay
# from agents.scr_ratio import SCR_RL_ratio
# from agents.scr_rl_addIter import SCR_RL_iter
# from agents.ER_RL_iter import ER_RL_iter
# from agents.unused.ER_RL_addIter import ER_RL_addIter
# from agents.unused.ER_RL_addIter_stop import ER_RL_addIter_stop
# from agents.unused.ER_RL_addIter_stop_new import ER_RL_addIter_stop_new
from agents.ER_dyna_iter import ER_dyna_iter
#from agents.ER_dyna_rnd import ER_dyna_rnd
from agents.ER_dyna_iter_aug import ER_dyna_iter_aug
#from agents.ER_dyna_iter_aug_only import ER_dyna_iter_aug_only
#from agents.unused.ER_dyna_iter_aug_dbpg import ER_dyna_iter_aug_dbpg
from agents.ER_dyna_iter_aug_dbpg_joint import ER_dyna_iter_aug_dbpg_joint
from utils.buffer.sc_retrieve import Match_retrieve
from utils.buffer.mem_match import MemMatch_retrieve



data_objects = {
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
    'core50': CORE50,
    'mini_imagenet': Mini_ImageNet,
    'openloris': OpenLORIS,
    'clrs25':CLRS25
}
agents = {
    'ER': ExperienceReplay,
    'ER_cl': ExperienceReplay_cl,
    "ER_cl_bandits":ExperienceReplay_cl_bandits,
    "ER_cl_bandits_ts":ExperienceReplay_cl_bandits_ts,
    "ER_batchsize":ExperienceReplay_batchsize,
    'ER_aug': ExperienceReplay_aug,
    "ER_ratio":ExperienceReplay_ratio,
    "ER_offline":ExperienceReplay_offline,
  #  "ER_RL_ratio":ER_RL_ratio,
#"ER_RL_iter":ER_RL_iter,
# "ER_RL_addIter":ER_RL_addIter,
#     "ER_RL_addIter_stop": ER_RL_addIter_stop,
#     "ER_RL_addIter_stop_new": ER_RL_addIter_stop_new,
#     "ER_dyna_rnd":ER_dyna_rnd,
    "ER_dyna_iter":ER_dyna_iter,
    "ER_dyna_iter_aug":ER_dyna_iter_aug,
    # "ER_dyna_iter_aug_only":ER_dyna_iter_aug_only,
    # "ER_dyna_iter_aug_dbpg":ER_dyna_iter_aug_dbpg,
    "ER_dyna_iter_aug_dbpg_joint":ER_dyna_iter_aug_dbpg_joint,
    #'RLER': RL_ExperienceReplay,
    #'LAMAML': LAMAML,
    'EWC': EWC_pp,
    'AGEM': AGEM,
    'CNDPM': Cndpm,
    'LWF': Lwf,
    'ICARL': Icarl,
    'GDUMB': Gdumb,
    'SCR': SupContrastReplay,
#     'SCR_RL_ratio':SCR_RL_ratio,
# 'SCR_RL_iter':SCR_RL_iter,
    #'SCR_META':SupContrastReplay_meta,
}

retrieve_methods = {
    'MIR': MIR_retrieve,
    'random': Random_retrieve,
    'ASER': ASER_retrieve,
    'match': Match_retrieve,
    'mem_match': MemMatch_retrieve,
    'RL': RL_retrieve

}

update_methods = {
    'random': Reservoir_update,
    'GSS': GSSGreedyUpdate,
    'ASER': ASER_update,
    'rt':Reservoir_update,
'rt2':Reservoir_update,
    'timestamp':Reservoir_update,
}



