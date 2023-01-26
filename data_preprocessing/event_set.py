import pandas as pd
import numpy as np
import vaex as vx
import pickle as pkl
import gc

from utility import *


class event_set:
    def __init__(self):
        
        self.grant = vx.open('data_preprocessing/grant_reduced.hdf5')
        self.cit = vx.open('data_preprocessing/edge_list.hdf5')
        
    def fit(self):
        
        ##################################
        ### NODE LIST TIME INFORMATION ###
        ##################################
        
        self.grant['num_id'] = vx.vrange(0,len(self.grant),dtype='int')
        self.grant['pub_year'] = self.grant['pubdate'].str.slice(0,4).astype('int')
        
        pub_date = self.grant['pubdate'].to_pandas_series()
        pub_date = pd.to_datetime(pub_date,format="%Y/%m/%d")
        starting_day = pd.Timestamp('1976-01-01')
        self.grant['pub_date_days'] = np.array((pub_date-starting_day).dt.days)


        self.grant.drop('pubdate',inplace=True)
        #self.grant.drop('owner',inplace=True)
        
        ######################################
        ### COMBINE EDGELIST WITH NODELIST ###
        ######################################

        #TODO: for i in self.grant.column_names:

        self.grant.rename('patnum','sender')
        self.grant.rename('num_id','sender_pos')
        self.grant.rename('pub_year','event_year')
        self.grant.rename('pub_date_days','event_day')
        self.grant.rename('ipc_count','ipc_count_sender')

        self.cit = self.cit.join(self.grant,on='sender',how='inner',allow_duplication=True)

        self.grant.rename('sender','receiver')
        self.grant.rename('sender_pos','receiver_pos')
        self.grant.rename('event_year','rec_pub_year')
        self.grant.rename('event_day','rec_pub_day')
        self.grant.rename('ipc_count_sender','ipc_count_receiver')

        self.cit = self.cit.join(self.grant,on='receiver',how='inner',allow_duplication=True)


        self.cit = remove_duplicates(self.cit,['sender_pos','receiver_pos'])
        del self.grant
        gc.collect()
        
        ##############################
        ### CITATION TEMPORAL LAG  ###
        ##############################
        
        self.cit['lag'] = self.cit['event_day']-self.cit['rec_pub_day']
        self.cit = self.cit[self.cit['lag']>=0]
        
        self.cit = self.cit.sort('event_day')
        self.cit = self.cit.to_pandas_df()
        gc.collect()
        
        #######################
        ### TEXT SIMILARITY ###
        #######################
        
        with open('data_preprocessing/embeddings.pkl', "rb") as input_file:
            emb = pkl.load(input_file)['embeddings']
            
        
        textual_similarity,_ = TexSim(sender_pos = self.cit['sender_pos'].to_numpy(),
                                      rec_pos = self.cit['receiver_pos'].to_numpy(),
                                      emb_matrix = emb,
                                      pub_date_y = self.cit['event_year'].to_numpy())
        
        self.cit['sim'] = textual_similarity
                
        del emb
        gc.collect()
        
        self.cit = vx.from_pandas(self.cit)
        
        ###############################################
        ### CUMULATIVE NUMBER OF CITATIONS RECEIVED ###
        ###############################################

        self.cit['tmp'] = np.repeat(1,len(self.cit))
        tmp_df = self.cit['receiver_pos','tmp'].to_pandas_df()
        cumu_cit_rec = tmp_df.groupby(['receiver_pos'])['tmp'].cumsum()

        self.cit.drop('tmp',inplace=True)
        self.cit['cumu_cit_rec']=np.array(cumu_cit_rec)

        del tmp_df,cumu_cit_rec
        gc.collect()

        #############################
        ### TIME FROM LAST EVENT  ###
        #############################

        rec = self.cit['receiver','event_day','lag'].to_pandas_df()
        tmp = rec.groupby(['receiver'])['event_day'].diff()
        rec['tfe']=tmp.fillna(rec['lag'])
        rec['tfe'] = rec['tfe'].astype('int64')
        
        self.cit['tfe']=np.array(rec['tfe'])

        del rec, tmp 
        gc.collect()
        
        #########################
        ### RECEIVER OUTDEGRE ###
        #########################

        #TODO: RECEIVER OUTDEGREE?
        
        #tmp = self.cit.groupby('sender',agg='count')
        #tmp.rename('sender','receiver')
        #tmp.rename('count','receiver_outd')
        
        #self.cit = self.cit.join(tmp, how='inner',on='receiver',allow_duplication=True)
        #gc.collect()
        #self.cit = self.cit.sort('event_day')
        
        ###################
        ### SAVING DATA ###
        ###################


        self.cit.export('data_preprocessing/event_set.hdf5',progress=True)
