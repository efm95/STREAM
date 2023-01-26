import pandas as pd
import numpy as np
import vaex as vx

from data_preprocessing.conversion_and_cleaning import *
from data_preprocessing.text_embedding import *
from data_preprocessing.event_set import *

from effects.sub_risk_set import *
from effects.rset_sampling import *
from effects.effects_creation import *

from model.model_selection import *

from utility import *

import gc

data_preprocessing:bool = True
model_selection:bool = True
model_fitting:bool = True

fit_repetition= 100

effects = 'rec_pub_year','rec_pub_year_2','lag','lag_2','sim','sim_2','cumu_cit_rec','cumu_cit_rec_2','tfe','tfe_2'


if __name__ == '__main__':
    
    if data_preprocessing:
        
        node_list_clean(grant ='data_preprocessing/grant_grant.csv',
                        ipc = 'data_preprocessing/grant_ipc.csv').fit()
        print('Node list cleaned')

        edge_list_clean(cit = 'data_preprocessing/grant_cite.csv').fit()
        print('Edge list cleaned')

        SBERT().fit()
        print('SBERT done')

        event_set().fit()
        print('Event set generated')
        
        sub_riskset().fit()
        print('Sub-risk-set generated')
        
        rset_sampling().fit()
        print('Risk set sampled')
        
        effects_creation().fit()
        print('Effect_created')
        
        
    if model_selection:
        
        df = vx.open('effects/effects.hdf5')[effects]
        X = df.sample(frac=1,replace=False,random_state=10091995)
    
        X['cumu_cit_rec'] = X['cumu_cit_rec'].log()
        X['cumu_cit_rec_2'] = X['cumu_cit_rec_2'].log()
    
        X['tfe'] = (X['tfe']+1).log()
        X['tfe_2'] = (X['tfe_2']+1).log()
    
        X = np.array(X.to_arrays()).transpose()    
        X = torch.from_numpy(X)
            
        grid = list(range(4,21,2))
        batch_size_grid = [2**10,
                           2**14,
                           2**18]
        
        dir_names=['model_selection1.pt',
                   'model_selection2.pt',
                   'model_selection3.pt']
        
        for i in range(len(batch_size_grid)):
            k_fold = REM_model_selection(X=X,
                                         df_grid=grid,
                                         batch_size=batch_size_grid[i],
                                         dir_name=dir_names[i])   
            gc.collect()          
        
    if model_fitting:
        df = vx.open('effects/effects.hdf5')[effects]
        splines_df = [9]*5
        coef_mat = torch.zeros((fit_repetition,sum(splines_df)))
        
        for rep in range(fit_repetition):
        
            X = df.sample(frac=1,replace=False,random_state=rep)
    
            X['cumu_cit_rec'] = X['cumu_cit_rec'].log()
            X['cumu_cit_rec_2'] = X['cumu_cit_rec_2'].log()
    
            X['tfe'] = (X['tfe']+1).log()
            X['tfe_2'] = (X['tfe_2']+1).log()
            
            X = np.array(X.to_arrays()).transpose()    
            X = torch.from_numpy(X)
            
            model = REM_sgd(splines_df = splines_df)
            model.fit(X=X,batch_size=2**16)
            
            coef_mat[rep,:] = model.get_coefs()
            
            gc.collect()
            
        torch.save(coef_mat,'coef_estim_rep100.pt')
        
        
