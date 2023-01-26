import pandas as pd 
import numpy as np 
import vaex as vx

from tqdm import tqdm

class rset_sampling:
    def __init__(self) -> None:
        self.event_set = vx.open('data_preprocessing/event_set.hdf5')
        self.sub_rec = vx.open('effects/sub_risk_set.hdf5').sort('rec_pub_day')
        
        with open('effects/sub_risk_set.npy','rb') as f:
            self.rset_cum_cit = np.load(f)
            self.rset_tfe = np.load(f)
            
    def fit(self):
        
        #self.sub_rec['index'] = vx.vrange(0,len(self.sub_rec),1,dtype='int')
        index = np.array(range(len(self.sub_rec)))
        
        day_count = self.event_set.groupby('event_day',agg='count').sort('event_day').to_pandas_df()
        sub_rec_day_count = self.sub_rec.groupby('rec_pub_day',agg='count').sort('rec_pub_day').to_pandas_df()

        #reduced_sub_rec = self.sub_rec['rec_pub_day','pos'].to_pandas_df()
        
        pos = np.array([],dtype=np.int64)
        cumu_cit = np.array([],dtype=np.int64)
        tfe = np.array([],dtype='int')
        event_day = np.array([])
        
        #rec_pub_day = reduced_sub_rec['rec_pub_day'].to_numpy()
        #rec_pub_pos = reduced_sub_rec['index'].to_numpy()
        
        pos_counter = 0
        
        for i in tqdm(range(len(day_count))):
            
            day = day_count.iloc[i,0]
            n_events = day_count.iloc[i,1]
            
            pos_counter += sub_rec_day_count.iloc[i,1]
            
            non_events = index[:pos_counter]
            #non_events = np.where(rec_pub_day<=day)[0]
            #non_events = reduced_sub_rec[reduced_sub_rec['rec_pub_day']<=day]['pos'].to_numpy()
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
        self.sub_rec.export('effects/risk_set.hdf5',progress=True)
        