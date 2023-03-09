import numpy as np
import torch
import vaex as vx

from model.bspline import *
from model.rem_logit import *
from model.rem_sgd import *

import gc
import logging
logging.basicConfig(format='%(asctime)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
    encoding="utf-8")

def REM_model_selection (X:torch.tensor,
                         df_grid:list,
                         batch_size:int = 2**17,
                         folds:int=6,
                         save:bool = True,
                         dir_name:str = 'model_selection.pt'):
    
    logging.info('Model selection')
    index_long = torch.tensor(range(len(X)))
    index = torch.tensor_split(index_long,folds)
    
    mask = torch.ones((len(index_long)),dtype=torch.bool)
    
    loss_mat = torch.zeros(len(index),len(df_grid))
    
    for d in tqdm(range(len(df_grid))):
        for rep in range(len(index)):
            
            logging.info(f'Degrees of freedom: {df_grid[d]} | Fold: {rep+1}/{len(index)}')
            index_val = index[rep]
            
            mask[index_val] = False
            
            X_val = X[index_val,:]
            X_train = X[mask,:]
            
            spline_df = [df_grid[d]]*(X.shape[1]//2)
            trainer = REM_sgd(spline_df=spline_df,lr=0.01)
            trainer.fit(X=X_train,batch_size=batch_size,verbose=False)
            
            val_loss = trainer.test_loss(X_val)
            
            loss_mat[rep,d] = val_loss
            
            mask[index_val] = True
        
    if save:
        torch.save(loss_mat,dir_name)
    
    return loss_mat 
        