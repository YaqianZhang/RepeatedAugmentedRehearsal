
#from models.resnet import Reduced_ResNet18
#import torch.nn as nn
from RL.pytorch_util import  build_mlp
import torch


PATH = "results/701/ER_random_random_NMC_testbatch100_RLmemIter_31_11_numRuns1_20_5000_cifar100_model"
model_dict = torch.load(PATH)

#nclass=100
#model = Reduced_ResNet18(nclass)
model = build_mlp(4,3,n_layers=2,size=32)


#optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

model.eval()

x = torch.zeros((1,4))
with torch.no_grad():
    y = model.forward(x)
    print(y)