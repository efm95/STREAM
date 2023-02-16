import pandas as pd
import numpy as np
import vaex as vx
import pickle as pkl
import gc

from utility import *

import logging
logging.basicConfig(format='%(asctime)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

class event_set:
    def __init__(self):
        
        self.grant = vx.open('data_preprocessing/grant_reduced.hdf5')
        self.cit = vx.open('data_preprocessing/edge_list.hdf5')
        
    def fit(self):
        
        ##################################
        ### NODE LIST TIME INFORMATION ###
        ##################################
        
        logging.info('Number od nodes: {}'.format(len(self.grant)))
        logging.info('Node list head \n {}'.format(self.grant.head()))
        
        
        ######################################
        ### COMBINE EDGELIST WITH NODELIST ###
        ######################################

        logging.info('Combining edge list with node list information')
        
        #TODO: for i in self.grant.column_names:
        
        self.grant.rename('patnum','sender')
        self.grant.rename('embedding_pos','sender_pos')
        self.grant.rename('pub_year','event_year')
        self.grant.rename('pub_date_days','event_day')

        self.cit = self.cit.join(self.grant,on='sender',how='inner',allow_duplication=True)

        self.grant.rename('sender','receiver')
        self.grant.rename('sender_pos','receiver_pos')
        self.grant.rename('event_year','rec_pub_year')
        self.grant.rename('event_day','rec_pub_day')

        self.cit = self.cit.join(self.grant,on='receiver',how='inner',allow_duplication=True)


        self.cit = remove_duplicates(self.cit,['sender_pos','receiver_pos'])
        
        gc.collect()
        logging.info('Number of event/edgees: {}'.format(len(self.cit)))
        logging.info('Event/edge list head \n {}'.format(self.cit.head()))
        
        ##########################
        ### JACCARD SIMILARITY ###
        ##########################
        
        logging.info('Computing Jaccard similarity across ipc classes')
        
        ipc= ipc_classes(ipc = 'data_preprocessing/grant_ipc.csv',
                         seq_len = 3)
        
        ipc.rename('patnum','sender')
        ipc.rename('classes','ipc_sender')
        
        self.cit['index'] = vx.vrange(0,len(self.cit),dtype='int')

        tmp = self.cit.copy()['sender','receiver','index']
        
        tmp = tmp.join(ipc,on='sender',how='inner',allow_duplication=True)
        
        ipc.rename('sender','receiver')
        ipc.rename('ipc_sender','ipc_receiver')
        
        tmp = tmp.join(ipc,on='receiver',how='inner',allow_duplication=True)
        
        sender_classes=tmp['ipc_sender'].to_numpy()
        receiver_classes = tmp['ipc_receiver'].to_numpy()
        
        logging.info('Computing Jac. similarity for each event')
        jac_sim = [jaccard_similarity(sender_classes[i],receiver_classes[i]) for i in tqdm(range(len(tmp)))]
        
        tmp['jac_sim'] = np.asanyarray(jac_sim)
        tmp.drop(['sender','receiver','ipc_sender','ipc_receiver'],inplace=True)
        
        jac_sim = tmp.sort('index')#.drop('index')['jac_sim'].to_numpy()
        
        self.cit = self.cit.join(tmp, on='index',allow_duplication=True, how='inner').drop('index')
        
        del tmp,sender_classes,receiver_classes
        gc.collect()
        logging.info('Number of event/edgees: {}'.format(len(self.cit)))
        logging.info('Event/edge list head \n {}'.format(self.cit.head()))
        
        ##############################
        ### CITATION TEMPORAL LAG  ###
        ##############################
        
        logging.info('Removing negative time-lags')
        self.cit['lag'] = self.cit['event_day']-self.cit['rec_pub_day']
        self.cit = self.cit[self.cit['lag']>=0]
        logging.info('Number of event/edgees: {}'.format(len(self.cit)))
        
        ###############
        ### SORTING ###
        ###############
        
        logging.info('Sorting events according to event day')
        self.cit = self.cit.sort('event_day')
        
        gc.collect()
        logging.info('Event/edge list head \n {}'.format(self.cit.head()))
        
        #######################
        ### TEXT SIMILARITY ###
        #######################
        
        logging.info('Computing textual similarity')
        with open('data_preprocessing/embeddings.pkl', "rb") as input_file:
            emb = pkl.load(input_file)['embeddings']
            
        
        textual_similarity = TexSim(sender_pos = self.cit['sender_pos'].to_numpy(),
                                    rec_pos = self.cit['receiver_pos'].to_numpy(),
                                    emb_matrix = emb,
                                    pub_date_y = self.cit['event_year'].to_numpy())
        
        
        #textual_sim = vx.from_arrays(pos=pos,sim = textual_similarity)
        
        
        self.cit['sim'] = textual_similarity['sim'].to_numpy()
        
        #self.cit['pos'] = vx.vrange(0,len(self.cit),dtype='int')
        #self.cit = self.cit.join(textual_sim,on='pos',how='inner',allow_duplication=False).drop('pos')
                
        del emb,textual_similarity
        gc.collect()
        
        logging.info('Event/edge list head \n {}'.format(self.cit.head()))
        
        #self.cit = vx.from_pandas(self.cit)
    
        ###############################################
        ### CUMULATIVE NUMBER OF CITATIONS RECEIVED ###
        ###############################################
        
        logging.info('Compuing cumulative number of citations received')
        #self.cit['tmp'] = np.repeat(1,len(self.cit))
        tmp = np.repeat(1,len(self.cit))
        
        #tmp_df = self.cit['receiver_pos','tmp'].to_pandas_df()
        tmp_df = pd.DataFrame({'receiver_pos':self.cit['receiver_pos'].to_numpy(),'tmp':tmp})
        cumu_cit_rec = tmp_df.groupby(['receiver_pos'])['tmp'].cumsum()

        #self.cit.drop('tmp',inplace=True)
        self.cit['cumu_cit_rec']=np.array(cumu_cit_rec)

        del tmp_df,cumu_cit_rec
        gc.collect()
        logging.info('Event/edge list head \n {}'.format(self.cit.head()))

        #############################
        ### TIME FROM LAST EVENT  ###
        #############################
        
        logging.info('Computing time from last event')
        rec = self.cit['receiver','event_day','lag'].to_pandas_df()
        tmp = rec.groupby(['receiver'])['event_day'].diff()
        rec['tfe']=tmp.fillna(rec['lag'])
        rec['tfe'] = rec['tfe'].astype('int64')
        
        self.cit['tfe']=np.array(rec['tfe'])

        del rec, tmp 
        gc.collect()
        logging.info('Event/edge list head \n {}'.format(self.cit.head()))
        
        #########################
        ### RECEIVER OUTDEGRE ###
        #########################
        
        logging.info('Computing receiver outdegree')
        tmp = self.cit.groupby('sender',agg='count')
        tmp.rename('sender','receiver')
        tmp.rename('count','receiver_outd')
        
        self.cit = self.cit.join(tmp, how='inner',on='receiver',allow_duplication=True)
        gc.collect()
        
        self.cit = self.cit.sort('event_day')
        
        logging.info('Number of event/edgees: {}'.format(len(self.cit)))
        logging.info('Event/edge list head \n {}'.format(self.cit.head()))
        
        ###################
        ### SAVING DATA ###
        ###################
        
        logging.info('Saving event set data to hdf5 format')
        self.cit.export('data_preprocessing/event_set.hdf5',progress=True)
