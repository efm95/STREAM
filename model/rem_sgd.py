import numpy as np
import torch
import torch.nn as nn

from model.bspline import * 
from model.rem_logit import *

from utility import *


class REM_sgd:
    def __init__(self,
                 spline_df:list,
                 spline_df_inter:list = None,
                 lr=0.01):
        self.spline_df = spline_df
        self.spline_df_inter = spline_df_inter
        self.device = device_identifier()

        if spline_df_inter is not None:
            self.model = REM_logit(p=sum(self.spline_df)+sum([self.spline_df_inter[i]*self.spline_df_inter[i+2] for i in range(0,len(self.spline_df_inter),4)])).to(self.device)
        else:
            self.model = REM_logit(p=sum(self.spline_df)).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.NLL = torch.nn.BCELoss()
    
    def fit(self,
            X:torch.tensor,
            X_inter:torch.tensor=None,
            batch_size:int = 2**17,
            epochs:int = 10,
            val_batch:int =0,
            verbose:bool = True):
                
        #X = bi-dimensional tensor with "event","non-event" column structure
        #n_obs = len(X)
        variables = int(X.shape[1]/2)
        self.bounds = torch.zeros(variables,2)
        for i in range(0,variables):
            self.bounds[i] = torch.tensor([torch.min(X[:,i*2]),
                                           torch.max(X[:,i*2])])
        
        if X_inter is not None:
            self.n_inter = int(X_inter.shape[1]/4)
            tot_eff = int(X_inter.shape[1]/2)
            self.inter_bounds = torch.zeros((tot_eff,2))
            for i in range(tot_eff):
                self.inter_bounds[i]=torch.tensor([torch.min(X_inter[:,i*2]),
                                              torch.max(X_inter[:,i*2])])
                
        
        X = X.split(batch_size)
        n_batches = len(X)
        
        if X_inter is not None:
            X_inter = X_inter.split(batch_size)
        
        #batch_val_pos = torch.randint(0,n_batches,(1,),generator=torch.manual_seed(val_batch_seed)).item()
        x_val = X[val_batch]
        x_val = spline_diff(x_val,df = self.spline_df,bounds=self.bounds)
        
        if X_inter is not None:
            x_val_inter = torch.tensor([[]]).reshape(batch_size,0)
            x_inter = X_inter[val_batch]
            for i in range(0,self.n_inter):
                x_tmp = x_inter[:,i*4:(i+1)*4]
                tmp = spline_diff_inter(X = x_tmp,
                                        df = list(self.spline_df_inter[i*4:(i+1)*4]),
                                        bounds=self.inter_bounds[i*2:(i+1)*2,:])
                x_val_inter = torch.column_stack((x_val_inter,tmp))
            
            x_val = torch.column_stack((x_val,x_val_inter))
        
        x_val = x_val.to(device=self.device)
        
        best_val_loss = torch.tensor(float('Inf')).item()
        best_batch = 0
        curr_batch = 0
        stopping_criterion=False
        
        self.model.train()
        for epoch in range(epochs):    
    
            for batch in range(n_batches):
                if batch == val_batch:
                    continue
                
                curr_batch += 1
                
                #Train step 
                self.model.train()
                x_tmp = X[i]
                x_tmp = spline_diff(x_tmp,
                                  df = self.spline_df,
                                  bounds=self.bounds)
                
                if X_inter is not None:
                    x_train_inter = torch.tensor([[]]).reshape(batch_size,0)
                    x_inter = X_inter[i]
                    for i in range(0,self.n_inter):
                        x_tmp_inter = x_inter[:,i*4:(i+1)*4]
                        tmp = spline_diff_inter(X = x_tmp_inter,
                                                df = list(self.spline_df_inter[i*4:(i+1)*4]),
                                                bounds=self.inter_bounds[i*2:(i+1)*2,:])
                        
                    
                        x_train_inter = torch.column_stack((x_train_inter,tmp))
                    
                    x_tmp = torch.column_stack((x_tmp,x_train_inter))   
                
                x_tmp = x_tmp.to(self.device)
                logit_out = self.model(x_tmp)
                loss = self.NLL(logit_out,torch.ones(len(x_tmp),1,device=self.device))
                perp = np.exp(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                #Validation step
                self.model.eval()
                logit_val = self.model(x_val)
                val_loss = self.NLL(logit_val,torch.ones(len(x_tmp),1,device=self.device)).item()
                
                if verbose:
                    print(f'Epoch: {epoch+1} | Batch {batch+1}/{n_batches} | batch loss = {np.round(loss.item(),4)} | val loss = {np.round(val_loss,4)} | Perplexity = {np.round(perp.item(),4)}')

                #Stpping criterion
                if np.round(val_loss,4) < np.round(best_val_loss,4):
                    best_val_loss = val_loss
                    best_batch += 1                    
                else:
                    if (curr_batch-best_batch)==10:
                        stopping_criterion = True
                        print(f'Iteration stopped at epoch {epoch +1}, batch {batch+1}/{n_batches} with train_loss {np.round(loss.item(),4)} and val. loss {np.round(val_loss,4)}')
                        break                 
                
            if stopping_criterion:
                break
            
    def test_loss(self,
                 X:torch.tensor,
                 X_inter:torch.tensor = None,
                 AIC:bool=True):
        
        X = spline_diff(X=X,df = self.spline_df,bounds=self.bounds)
        
        if X_inter is not None:
            x_test_inter = torch.tensor([[]]).reshape(len(X),0)
            for i in range(0,self.n_inter):
                x_tmp_inter = X_inter[:,i*4:(i+1)*4]
                tmp = spline_diff_inter(X = x_tmp_inter,
                                        df = list(self.spline_df_inter[i*4:(i+1)*4]),
                                        bounds=self.inter_bounds[i*2:(i+1)*2,:])
                
                x_test_inter = torch.column_stack((x_test_inter,tmp))
            
            X = torch.column_stack((X,x_test_inter))
            
        X = X.to(self.device)
        n = len(X)
        
        self.model.eval()
        logit_out =self.model(X)
        loss = self.NLL(logit_out,torch.ones(len(X),1,device=self.device)).item()
        
        return loss
    
    def get_coefs(self):
        return self.model.linear.weight.detach().squeeze(0)
