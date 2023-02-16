import pandas as pd 
import numpy as np
import vaex as vx

import torch 

from model.bspline import * 
from utility import *

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(format='%(asctime)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

class fitted_values:
    def __init__(self,
                 event_effect:torch.tensor,
                 non_event_effects:torch.tensor):
        
        logging.info('Fitted values')
        self.event_effects = event_effect
        self.non_event_effects = non_event_effects
        
        
        self.variables = self.event_effects.shape[1]
        
        self.bounds = torch.zeros(self.variables,2)
        for var in range(0,self.variables):
            self.bounds[var] = torch.tensor([torch.min(self.event_effects[:,var]),
                                             torch.max(self.event_effects[:,var])])
            
        
    def fit(self,
            coefficients:torch.tensor,
            batch_size:int = 2**17):
        
        spline_df = len(coefficients)//self.event_effects.shape[1]
        logging.info('Batch splitting')
        X = self.event_effects.split(batch_size)
        Y = self.non_event_effects.split(batch_size)
        
        n_batches = len(X)
        
        output_events= torch.tensor([]) 
        output_non_events = torch.tensor([])
        
        logging.info('Computing fitted values per batch')
        for batch in tqdm(range(n_batches)):
            
            X_tmp = X[batch]
            Y_tmp = Y[batch]
            
            event_spline = torch.tensor([]).reshape(len(X_tmp),0)
            non_event_spline = torch.tensor([]).reshape(len(Y_tmp),0)
            
            for var in range(0,self.variables):
        
                event_eff = Bspline(x=X_tmp[:,var],
                                    df = spline_df,
                                    lower_bound=self.bounds[var][0],
                                    upper_bound=self.bounds[var][1]).fit()
                
                event_spline = torch.column_stack((event_spline,event_eff))
                
                non_event_eff = Bspline(x=Y_tmp[:,var],
                                        df = spline_df,
                                        lower_bound=self.bounds[var][0],
                                        upper_bound=self.bounds[var][1]).fit()
                
                non_event_spline = torch.column_stack((non_event_spline,non_event_eff))
        
            output_x_tmp = event_spline@coefficients        
            output_y_tmp = non_event_spline@coefficients
            
            output_events = torch.cat((output_events,output_x_tmp))
            output_non_events = torch.cat((output_non_events,output_y_tmp))
            
        return output_events,output_non_events
        
        
def baseline(effect_list:vx.DataFrame,
             event_list:vx.DataFrame,
             fitted_values_event:torch.tensor,
             fitted_values_non_event:torch.tensor):
    
    logging.info('Cumulative aseline hazard')
    denom = torch.exp(fitted_values_event) + torch.exp(fitted_values_non_event)
    
    receivers = remove_duplicates(event_list,grouping_cols=['receiver_pos','rec_pub_day'])
    
    event_count = effect_list.groupby(by='event_day',agg='count')
    event_count.rename('count','day_count')
    event_count = event_count.sort(by='event_day')
    event_count['event_day'] = event_count['event_day'].astype('int64')
    event_count = event_count.to_pandas_df()
    
    logging.info('Computing number of events at risk for each observed event period')
    events_at_risk = receivers.groupby(by = 'rec_pub_day',agg='count').sort('rec_pub_day')
    at_risk = np.array([],dtype=np.int32)
    for i in tqdm(range(len(event_count))):
        event_day = event_count['event_day'][i]
        at_risk = np.append(at_risk,events_at_risk[events_at_risk['rec_pub_day']<event_day]['count'].sum())
    
    event_count['n_at_risk'] = at_risk
    effect_list = effect_list.join(vx.from_pandas(event_count),on='event_day',how='inner',allow_duplication=True)
    #effect_list['denom'] = denom.squeeze(1).numpy()
    
    logging.info('Computing hazard rate and cumulative hazard')

    denom_corrected = denom.numpy()*(effect_list['n_at_risk'].to_numpy()/2)
    hazard = 1/denom_corrected
    effect_list['haz'] = hazard
    out = effect_list['event_day','haz'].groupby(by='event_day',agg='sum').sort('event_day').to_pandas_df()
    out['H'] = out['haz_sum'].to_numpy().cumsum()
    
    return out

