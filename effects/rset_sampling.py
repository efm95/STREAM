import pandas as pd 
import numpy as np 
import vaex as vx

from tqdm import tqdm

import logging
logging.basicConfig(format='%(asctime)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
    filename="log.txt")


class rset_sampling:
    def __init__(self):
        logging.info('Initialize risk-set sampling')
        self.event_set = vx.open('data_preprocessing/event_set.hdf5')
        self.sub_rec = vx.open('effects/sub_risk_set.hdf5')#.sort('rec_pub_day')
        
        with open('effects/sub_risk_set.npy','rb') as f:
            self.rset_cum_cit = np.load(f)
            self.rset_tfe = np.load(f)
            
    def fit(self):
        
        index = np.array(range(len(self.sub_rec)))
        
        day_count = self.event_set.groupby('event_day',agg='count').sort('event_day').to_pandas_df()
        sub_rec_day_count = self.sub_rec.groupby('rec_pub_day',agg='count').sort('rec_pub_day').to_pandas_df()

        
        pos = np.array([],dtype=np.int64)
        cumu_cit = np.array([],dtype=np.int64)
        tfe = np.array([],dtype='int')
        event_day = np.array([])
        
        pos_counter = 0
        
        logging.info('Start sampling')
        
        for i in tqdm(range(len(day_count))):
            
            day = day_count.iloc[i,0]
            n_events = day_count.iloc[i,1]
            
            pos_counter += sub_rec_day_count.iloc[i,1]
            
            non_events = index[:pos_counter]
            samp_pos = np.random.choice(a=non_events,size=n_events,replace=True)
            
            event_day = np.append(event_day,np.repeat(day,n_events))     
            
            pos = np.append(pos,samp_pos)
            cumu_cit = np.append(cumu_cit,self.rset_cum_cit[samp_pos,i])
            tfe = np.append(tfe,self.rset_tfe[samp_pos,i])
            
        
        self.sub_rec = self.sub_rec.to_pandas_df()
        self.sub_rec = self.sub_rec.iloc[pos,:]
        
        self.sub_rec['cumu_cit'] = cumu_cit
        
        #Correcting time from event (tfe)
        pos = np.where(tfe==0)[0]
        tfe[pos] = event_day[pos]-self.sub_rec['rec_pub_day'].to_numpy()[pos]
        self.sub_rec['tfe'] = tfe

        self.sub_rec = vx.from_pandas(self.sub_rec)
        logging.info('Sampled risk-set \n {}'.format(self.sub_rec.head()))
        
        self.sub_rec.export('effects/risk_set.hdf5',progress=True)
        
