import torch
import torch.nn as nn

class OneStageContrastDLRdm(nn.Module):
    def __init__(self, dlfeanum, rdmfeanum, dimp):
        super(OneStageContrastDLRdm, self).__init__()
        self.dlfeanum = dlfeanum
        self.rdmfeanum = rdmfeanum
        self.dimp = dimp

        # Define the layers of the model
        self.dl_layer = nn.Linear(dlfeanum, dimp)
        self.rdm_layer = nn.Linear(rdmfeanum, dimp)

    def forward(self, dlfea, rdmfea):
        dl_x = self.dl_layer(dlfea)
        rdm_x = self.rdm_layer(rdmfea)
        return dl_x, rdm_x