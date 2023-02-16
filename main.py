import pandas as pd
import numpy as np
import vaex as vx

import warnings
warnings.filterwarnings("ignore")

from data_preprocessing.conversion_and_cleaning import *
from data_preprocessing.text_embedding import *
from data_preprocessing.event_set import *

from effects.sub_risk_set import *
from effects.rset_sampling import *
from effects.effects_creation import *

from model.model_selection import *
from model.rem_logit import *
from model.rem_sgd import *

from utility import *

import gc

import logging
logging.basicConfig(format='%(asctime)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
    filename="rem.log",
    encoding="utf-8")


data_preprocessing:bool = False
model_selection:bool = False
model_fitting:bool = False
model_fitting_rep:bool = True

fit_repetition= 100

effects = 'rec_pub_year','rec_pub_year_2','lag','lag_2','sim','sim_2','jac_sim','jac_sim_2','cumu_cit_rec','cumu_cit_rec_2','tfe','tfe_2','rec_outd','rec_outd_2'


if __name__ == '__main__':
    
    if data_preprocessing:
        
        logging.info('Data preprocessing')
        
        node_list_clean(grant ='data_preprocessing/grant_grant.csv').fit()
        logging.info('Node list cleaned')

        edge_list_clean(cit = 'data_preprocessing/grant_cite.csv').fit()
        print('Edge list cleaned')
        logging.info('Edge list cleaned')
        
        #SBERT().fit()                              #Run on server with GPU node --> currently there is no mps support, only cuda
        #logging.info('SBERT embeddings computed')

        event_set().fit()
        logging.info('Event set generated')
        
        sub_riskset().fit()
        logging.info('Sub-risk-set generated')
        
        rset_sampling().fit()
        logging.info('Risk set sampled')
        
        effects_creation().fit()
        logging.info('Effects created')
        
        
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
        
    if model_fitting:
        df = vx.open('effects/effects.hdf5')[effects]
        
        splines_degrees = [10]*(len(effects)//2)
        coef_mat = torch.zeros((fit_repetition,sum(splines_degrees)))
        
        for rep in range(fit_repetition):
            
            logging.info('Repetition: {}'.format(rep+1))
            X = df.sample(frac=1,replace=False,random_state=rep)
    
            X['cumu_cit_rec'] = X['cumu_cit_rec'].log()
            X['cumu_cit_rec_2'] = X['cumu_cit_rec_2'].log()
    
            X['tfe'] = (X['tfe']+1).log()
            X['tfe_2'] = (X['tfe_2']+1).log()
            
            X['rec_outd'] = X['rec_outd'].log()
            X['rec_outd_2'] = X['rec_outd_2'].log()
            
            X = np.array(X.to_arrays()).transpose()    
            X = torch.from_numpy(X)
            
            model = REM_sgd(spline_df = splines_degrees)
            model.fit(X=X,
                      batch_size=2**15,
                      verbose=False)
            
            coef_mat[rep,:] = model.get_coefs()
            
            gc.collect()
            
        torch.save(coef_mat,'coef_estim/rep100_batches215_10df_final.pt')
        
        
    if model_fitting_rep: 
        n_sub = 20
        n_fit = 50
        splines_degrees = [10]*(len(effects)//2)
        coef_mat = torch.zeros((n_sub*n_fit,sum(splines_degrees)))#torch.tensor([])
        
        seed = 0
        
        for sub in range(n_sub):
            
            sub_riskset().fit(seed=seed)
            logging.info('Sub-risk-set generated')
        
            rset_sampling().fit()
            logging.info('Risk set sampled')
        
            effects_creation().fit()
            logging.info('Effects created')
            
            df = vx.open('effects/effects.hdf5')[effects]
                            
            for fit in range(n_fit):
                
                logging.info(f'Sub-risk-set: {sub+1}/{n_sub} | fit:{fit+1}/{n_fit}')
                
                X = df.sample(frac=1,replace=False,random_state=seed)
    
                X['cumu_cit_rec'] = X['cumu_cit_rec'].log()
                X['cumu_cit_rec_2'] = X['cumu_cit_rec_2'].log()
    
                X['tfe'] = (X['tfe']+1).log()
                X['tfe_2'] = (X['tfe_2']+1).log()
            
                X['rec_outd'] = X['rec_outd'].log()
                X['rec_outd_2'] = X['rec_outd_2'].log()
            
                X = np.array(X.to_arrays()).transpose()    
                X = torch.from_numpy(X)
            
                model = REM_sgd(spline_df = splines_degrees)
                model.fit(X=X,
                          batch_size=2**14,
                          verbose=False)
            
                coef_mat[seed,:] = model.get_coefs()
                #coef_mat = torch.row_stack((coef_mat,model.get_coefs()))
                seed +=1
                gc.collect()
        
        torch.save(coef_mat,'coef_estim/rem-50fit-50sub.pt')
