import pandas as pd
import numpy as np
import vaex as vx 
from tqdm import tqdm
import gc

from utility import *

import logging
logging.basicConfig(format='%(asctime)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

class sub_riskset:
    def __init__(self):
        logging.info('Initialize creation of sub-risk-set')
        self.df = vx.open('data_preprocessing/event_set.hdf5')
      
        
    def fit(self,
            n_samples:int = 10,
            seed: int = 10091995):
        
        dates = self.df['rec_pub_day'].unique() 
        dates.sort()
        
        #unique_rec = self.df['receiver','rec_pub_year','rec_pub_day','receiver_pos']
        logging.info('Filtering for unique receivers')
        unique_rec = self.df.copy()['receiver','rec_pub_year','rec_pub_day','receiver_pos','receiver_outd']
        unique_rec = remove_duplicates(unique_rec,['receiver','rec_pub_day'])
        #unique_rec['index']=vx.vrange(0,len(unique_rec),1,dtype='int')
        
        logging.info('Number of events at risk: {}'.format(len(unique_rec)))
        logging.info('Events a risk \n {}'.format(unique_rec.head()))
                
        day_count = unique_rec.groupby('rec_pub_day',agg='count')
        day_count = day_count.sort('rec_pub_day').to_pandas_df()
        day_count['rec_pub_day'] = day_count['rec_pub_day'].astype('int64')
        
        #Sampling patents for each publication date
        logging.info('Sampling events for each issue day')
        over_rec = vx.from_pandas(day_count[day_count['count'] > n_samples])
        under_rec = vx.from_pandas(day_count[day_count['count'] <= n_samples])
        
        over_rec = unique_rec.join(over_rec,on='rec_pub_day',how='inner',allow_duplication=True).drop('count').to_pandas_df()
        under_rec = unique_rec.join(under_rec,on='rec_pub_day',how='inner',allow_duplication=True).drop('count').to_pandas_df()
        
        over_rec = over_rec.groupby('rec_pub_day').sample(n_samples,replace=False,random_state=seed)
        under_rec = under_rec.groupby('rec_pub_day').sample(1,replace=False,random_state=seed)
        
        unique_rec= pd.concat([over_rec,under_rec],ignore_index=True,axis=0).sort_values(by='rec_pub_day')
        
        logging.info('Events in sub-risk-set: {}'.format(len(unique_rec)))
        logging.info('Sub-risk-set \n {}'.format(unique_rec.head()))
        
        unique_rec_tmp = unique_rec.copy()
        unique_rec_tmp['range'] = range(len(unique_rec_tmp)) 
        unique_rec_tmp = vx.from_pandas(unique_rec_tmp[['receiver','range']])

        reduced_df = self.df.join(unique_rec_tmp,on='receiver',how='inner',allow_duplication=True).drop('range')
        
        del unique_rec_tmp
        reduced_df = reduced_df['event_day','receiver','receiver_pos','cumu_cit_rec','tfe'].to_pandas_df()
        #reduced_df = reduced_df.sort('event_day')
        
        gc.collect()
        
        event_days = self.df['event_day'].unique()
        event_days.sort()
        
        event_pos = dict(zip(event_days,range(len(event_days))))
        pos_event = dict(zip(range(len(event_days)),event_days))
        
        risk_set_cum_cit = np.zeros((len(unique_rec),len(event_days)))
        risk_set_tfe = np.zeros((len(unique_rec),len(event_days)))
        
        logging.info('Computing time-varyinig effects')
        
        for i in tqdm(range(len(unique_rec))): #for every sampled received patent
    
            #tmp_rec = unique_rec.iloc[i,0]
            tmp_rec = unique_rec.iloc[i,3]
            #tmp_events = self.df[self.df['receiver']==tmp_rec]#.to_pandas_df()
            tmp_events = reduced_df[reduced_df['receiver_pos']==tmp_rec]#.to_pandas_df()
    
            event_day = tmp_events['event_day'].to_numpy()
            pos = [event_pos[i] for i in event_day]
    
            #Cumualtive sum
            #cumsum = range(1,len(tmp_events)+1)
            cumsum = tmp_events['cumu_cit_rec'].to_numpy()
            cumsum = np.append(cumsum,cumsum[-1]+1)
    
            #Time from last event
            tfe = tmp_events['tfe'].to_numpy()
    
            #Storing cumulative number of citations
            count=0
            for j in range(len(event_days)):
                risk_set_cum_cit[i,j] = cumsum[count]
                if count<len(pos):
                    if j == pos[count]:
                        count +=1
            
            #Storing time from last event
            risk_set_tfe[i,pos[0]] = tfe[0]
            cumulative_gap = 0
            for j in range(pos[0]+1,len(event_days)):
                gap = pos_event[j]-pos_event[j-1]
                cumulative_gap+=gap
                risk_set_tfe[i,j] = cumulative_gap
                if j in pos:
                    cumulative_gap = 0          
                
            #gc.collect()
        
        logging.info('Save sub-risk-set and time-varying effects')
        unique_rec = vx.from_pandas(unique_rec)
        unique_rec.export('effects/sub_risk_set.hdf5',progress=True)
        with open('effects/sub_risk_set.npy', 'wb') as f:
            np.save(f, risk_set_cum_cit)
            np.save(f, risk_set_tfe)

