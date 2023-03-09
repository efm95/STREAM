import pandas as pd
import numpy as np
import vaex as vx
import pickle as pkl
import gc

from utility import *

import logging
logging.basicConfig(format='%(asctime)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
    filename="log.txt")

class effects_creation:
    def __init__(self):
        
        logging.info('Initialize effects-creation routine')
        self.df = vx.open('data_preprocessing/event_set.hdf5').sort('event_day')
        self.rset = vx.open('effects/risk_set.hdf5')
        
    def fit(self):
        
        #################################
        ### EVENT AND PUBLICATION DAY ###
        #################################
        
        logging.info('Event and publication day')
        
        event_day = self.df['event_day'].to_numpy()
        rec_pub_day = self.df['rec_pub_day'].to_numpy()
        rec_pub_day_2 = self.rset['rec_pub_day'].to_numpy()
        
        ########################
        ### PUBLICATION YEAR ###
        ########################
        
        logging.info('Publication year')
        
        rec_pub_year = self.df['rec_pub_year'].to_numpy()
        rec_pub_year_2 = self.rset['rec_pub_year'].to_numpy()
        
        ###########
        ### LAG ###
        ###########
        
        logging.info('Time lag')
        
        lag = self.df['lag'].to_numpy()
        lag_2 = event_day - rec_pub_day_2
        
        ####################
        ### JACCARD IPC #### 
        ####################
        
        logging.info('Jaccard similarity on ipc class')
        jac_sim = self.df['jac_sim'].to_numpy()

        ipc= ipc_classes(ipc = 'data_preprocessing/grant_ipc.csv',seq_len = 3)
        
        ipc.rename('patnum','sender')
        ipc.rename('classes','ipc_sender')
        
        #self.df['index'] = vx.vrange(0,len(self.df),dtype='int')
        tmp = self.df['sender','receiver'].copy()
        tmp['receiver'] = self.rset['receiver'].to_numpy()
        tmp['index'] = vx.vrange(0,len(tmp),dtype='int')
        
        tmp = tmp.join(ipc,on='sender',how='inner',allow_duplication=True)
        
        ipc.rename('sender','receiver')
        ipc.rename('ipc_sender','ipc_receiver')
        
        tmp = tmp.join(ipc,on='receiver',how='inner',allow_duplication=True)
        
        sender_classes=tmp['ipc_sender'].to_numpy()
        receiver_classes = tmp['ipc_receiver'].to_numpy()
        
        logging.info('Computing Jac. similarity for each sender-non-receiver')
        jac_sim_2 = [jaccard_similarity(sender_classes[i],receiver_classes[i]) for i in tqdm(range(len(tmp)))]
        
        tmp['jac_sim_2'] = np.asanyarray(jac_sim_2)
        tmp.drop(['sender','receiver','ipc_sender','ipc_receiver'],inplace=True)
        
        jac_sim_2 = tmp.sort('index')['jac_sim_2'].to_numpy()
                
        del tmp,sender_classes,receiver_classes
        gc.collect()
        
        
        #ipc_count = self.df['ipc_count_receiver'].to_numpy()
        #ipc_count_2 = self.rset['ipc_count_receiver'].to_numpy()
        
        ##########################
        ### RECEIVER OUTDEGREE ### 
        ##########################
        
        logging.info('Receiver outdegree')

        rec_outd = self.df['receiver_outd'].to_numpy()
        rec_outd_2 = self.rset['receiver_outd'].to_numpy()
        
        ###############################################
        ### CUMULATIVE NUMBER OF CITATIONS RECEIVED ###
        ###############################################
        
        logging.info('Cumulative number of citations received')
        
        cumu_cit_rec = self.df['cumu_cit_rec'].to_numpy()
        cumu_cit_rec_2 = self.rset['cumu_cit'].to_numpy()
        
        ###################################
        ### TIME FROM PREVIOUS CITATION ###
        ###################################
        
        logging.info('Time from last event')
        
        tfe = self.df['tfe'].to_numpy()
        tfe_2 = self.rset['tfe'].to_numpy()
        
        ##################
        ### SIMILARITY ###
        ##################
        
        logging.info('Textal similarity')
        
        sim = self.df['sim'].to_numpy()
        
        logging.info('Computing textal similarity on sender and non-receiver')
        
        with open('data_preprocessing/embeddings.pkl','rb') as input_file:
            emb = pkl.load(input_file)['embeddings']
            
        #sim_2,pos = TexSim(sender_pos = self.df['sender_pos'].to_numpy().astype('int'),
        #                 rec_pos = self.rset['receiver_pos'].to_numpy().astype('int'),
        #                 emb_matrix = emb,
        #                 pub_date_y = self.df['event_year'].to_numpy().astype('int'))
        
        sim_2 = TexSim(sender_pos = self.df['sender_pos'].to_numpy().astype('int'),
                         rec_pos = self.rset['receiver_pos'].to_numpy().astype('int'),
                         emb_matrix = emb,
                         pub_date_y = self.df['event_year'].to_numpy().astype('int'))
        
        
        sim_2 = sim_2['sim'].to_numpy()
        #textual_sim_2 = vx.from_arrays(pos = pos,
        #                               sim_2 = sim_2)
        #
            
        del emb
        
        ##############
        ### SAVING ###
        ##############
        
        logging.info('Collecting every effect')
        
        self.effects = vx.from_arrays(event_day = event_day,
                                      
                                      rec_pub_day = rec_pub_day,
                                      rec_pub_year = rec_pub_year,
                                      
                                      rec_pub_day_2 = rec_pub_day_2,
                                      rec_pub_year_2 = rec_pub_year_2,
                                      
                                      lag = lag,
                                      lag_2 = lag_2,

                                      jac_sim = jac_sim,
                                      jac_sim_2 = jac_sim_2,
                                      
                                      cumu_cit_rec = cumu_cit_rec,
                                      cumu_cit_rec_2 = cumu_cit_rec_2,
                                      
                                      rec_outd = rec_outd,
                                      rec_outd_2 = rec_outd_2,
                                      
                                      tfe = tfe,
                                      tfe_2 = tfe_2,
                                      
                                      sim = sim,
                                      sim_2 = sim_2)
        
        assert len(self.effects[self.effects['lag']<0])==0
        assert len(self.effects[self.effects['lag_2']<0])==0
        
        logging.info('Effect matrix \n {}'.format(self.effects.head()))
        
        self.effects.export('effects/effects.hdf5',progress=True)

