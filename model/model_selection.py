import numpy as np
import torch
import vaex as vx

from model.bspline import *
from model.rem_logit import *
from model.rem_sgd import *

import gc

def REM_model_selection (X:torch.tensor,
                         df_grid:list,
                         batch_size:int = 2**17,
                         folds:int=6,
                         save:bool = True,
                         AIC:bool = False,
                         dir_name:str = 'model_selection.pt'):
    
    index_long = torch.tensor(range(len(X)))
    index = torch.tensor_split(index_long,folds)
    
    loss_mat = torch.zeros(len(index),len(df_grid))
    
    for d in range(len(df_grid)):
        for rep in range(len(index)):
            index_val = index[rep]
            mask = torch.ones((len(index_long)),dtype=torch.bool)
            mask[index_val] = False
            
            X_val = X[index_val,:]
            X_train = X[mask,:]
            
            spline_df = [df_grid[d]]*int(X.shape[1]/2)
            trainer = REM_sgd(spline_df=spline_df,lr=0.01)
            trainer.fit(X=X_train,batch_size=batch_size)
            
            val_loss = trainer.test_loss_AIC(X_val,
                                              AIC=AIC)
            
            loss_mat[rep,d] = val_loss
        
    if save:
        torch.save(loss_mat,dir_name)
    
    return loss_mat 
        