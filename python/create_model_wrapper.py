import MixtureOfExpertsAdvanced
from MixtureOfExpertsDNNFast import WrapperForDataParallel
import torch
import sys

model = torch.load(sys.argv[1], map_location='cpu')
dnn = model.module.dnn
wrapper = MixtureOfExpertsAdvanced.createMoEFullMergedAdvancedModelWrapper(dnn)
torch.save(wrapper, sys.argv[2])
