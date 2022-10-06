import numpy as np
import torch
import torch.nn as nn
from bspline import * 

class REM_logit(nn.Module):
    def __init__(self,
                 p:int):
        super(REM_logit,self).__init__()
        self.p = p
        self.linear = nn.Linear(self.p,1,bias=False)
        
    def forward(self,input):
        logit_out = torch.sigmoid(self.linear(input))
        
        return logit_out

class REM_sgd:
    def __init__(self,
                 spline_df:list,
                 lr):
        self.spline_df = spline_df
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = REM_logit(p=sum(self.spline_df)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.NLL = torch.nn.BCELoss()

        #self.val_loss = 0
        #self.loss = 0
        #self.negative_lkl = 0
        #self.AIC = 0
        #self.BIC = 0
    
    def fit(self,
            X:torch.tensor,
            batch_size:int = 2**17,
            epochs:int = 10,
            val_batch_seed:int =10091995):
        
        assert int(X.shape[1]/2)==len(self.spline_df), "Input shape and nuber of degrees of freedom for spline differ"
        
        #X = bi-dimensional tensor with "event","non-event" structure per column
        n_obs = len(X)
        variables = int(X.shape[1]/2)
        self.bounds = torch.zeros(variables,2)
        for i in range(0,variables):
            self.bounds[i] = torch.tensor([torch.min(X[:,i*2]),
                                           torch.max(X[:,i*2])])
        
        
        X = X.split(batch_size)
        n_batches = len(X)
        
        batch_val_pos = torch.randint(0,n_batches,(1,),generator=torch.manual_seed(val_batch_seed)).item()
        x_val = X[batch_val_pos]
        x_val = spline_diff(x_val,df = self.spline_df,bounds=self.bounds)
        x_val = x_val.to(device=self.device)
        
        best_val_loss = torch.tensor(float('Inf')).item()
        best_batch = -1
        curr_batch = -1
        stopping_criterion=False
        
        self.model.train()
        for epoch in range(epochs):    
    
            for batch in range(n_batches):
                if batch == batch_val_pos:
                    continue
                
                curr_batch += 1
                
                #Train step 
                self.model.train()
                x_tmp = X[i]
                x_tmp = spline_diff(x_tmp,
                                  df = self.spline_df,
                                  bounds=self.bounds)
                x_tmp = x_tmp.to(self.device)
                logit_out =self.model(x_tmp)
                loss = self.NLL(logit_out,torch.ones(len(x_tmp),1,device=self.device))
                perp = np.exp(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                #Validation step
                self.model.eval()
                logit_val = self.model(x_val)
                val_loss = self.NLL(logit_val,torch.ones(len(x_tmp),1,device=self.device)).item()
                
                print(f'Epoch: {epoch+1} | Batch {batch+1}/{n_batches} | batch loss = {np.round(loss.item(),4)} | val loss = {np.round(val_loss,4)} | Perplexity = {np.round(perp.item(),4)}')

                #Stpping criterion
                if np.round(val_loss,4) < np.round(best_val_loss,4):
                    best_val_loss = val_loss
                    best_batch += 1                    
                else:
                    if (curr_batch-best_batch)==10:
                        stopping_criterion = True
                        self.val_loss = best_val_loss
                        self.loss = loss.item()
                        #self.negative_lkl = (best_val_loss*batch_size)+(loss.item()*batch_size*n_batches)
                        #self.AIC = 2*len(self.model.linear.weight.detach().squeeze(0))-2*self.negative_lkl
                        #self.BIC = len(self.model.linear.weight.detach().squeeze(0))*np.log(n_obs)-(2*self.negative_lkl)
                        print(f'Iteration stopped at epoch {epoch +1}, batch {batch+1} with train_loss {np.round(loss.item(),4)} and val. loss {np.round(val_loss,4)}')
                        break                 
                
            if stopping_criterion:
                break
    
    def test_loss(self,
                  X:torch.tensor):
        
        X = spline_diff(X=X,df = self.spline_df,bounds=self.bounds)
        X = X.to(self.device)
        
        self.model.eval()
        logit_out =self.model(X)
        loss = self.NLL(logit_out,torch.ones(len(X),1,device=self.device)).item()
        
        return loss
        