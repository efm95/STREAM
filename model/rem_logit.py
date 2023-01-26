import torch
import torch.nn as nn

class REM_logit(nn.Module):

    def __init__(self,
                 p:int):
        super(REM_logit,self).__init__()
        self.p = p
        self.linear = nn.Linear(self.p,1,bias=False)
        
    def forward(self,input):
        logit_out = torch.sigmoid(self.linear(input))
        
        return logit_out
    