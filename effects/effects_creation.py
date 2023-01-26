import pandas as pd 
import numpy as np
import vaex as vx
import pickle as pkl

from utility import *

class effects_creation:
    def __init__(self):
        self.df = vx.open('event_set.hdf5').sort('event_day')
        self.rset = vx.open('risk_set.hdf5')
        
    def fit(self):
        
        #################################
        ### EVENT AND PUBLICATION DAY ###
        #################################
        
        event_day = self.df['event_day'].to_numpy()
        rec_pub_day = self.df['rec_pub_day'].to_numpy()
        rec_pub_day_2 = self.rset['rec_pub_day'].to_numpy()
        
        ########################
        ### PUBLICATION YEAR ###
        ########################
        
        rec_pub_year = self.df['rec_pub_year'].to_numpy()
        rec_pub_year_2 = self.rset['rec_pub_year'].to_numpy()
        
        ###########
        ### LAG ###
        ###########
        
        lag = self.df['lag'].to_numpy()
        lag_2 = event_day - rec_pub_day_2
        
        #################
        ### IPC COUNT ### TODO: SISTEMARE IPC COUNT 
        #################
        
        ##########################
        ### RECEIVER OUTDEGREE ### TODO: SISTEMARE 
        ##########################
        
        ###############################################
        ### CUMULATIVE NUMBER OF CITATIONS RECEIVED ###
        ###############################################
        
        cumu_cit_rec = self.df['cumu_cit_rec'].to_numpy()
        cumu_cit_rec_2 = self.rec['cumu_cit_rec'].to_numpy()
        
        ###################################
        ### TIME FROM PREVIOUS CITATION ###
        ###################################
        
        tfe = self.df['tfe'].to_numpy()
        tfe_2 = self.rec['tfe'].to_numpy()
        
        ##################
        ### SIMILARITY ###
        ##################
        
        sim = self.df['sim'].to_numpy()
        
        with open('data_preprocessing/embeddings.pkl','rb') as input_file:
            emb = pkl.load(input_file)['embeddings']
            
        sim_2,_ = TexSim(sender_pos = self.df['sender_pos'].to_numpy().astype('int'),
                         rec_pos = self.df['receiver_pos'].to_numpy().astype('int'),
                         emb_matrix = emb,
                         pub_date_y = self.df['event_year'].to_numpy().astype('int'))
        
        
        ##############
        ### SAVING ###
        ##############
        
        self.effects = vx.from_arrays(event_day = event_day,
                                      
                                      rec_pub_day = rec_pub_day,
                                      rec_pub_year = rec_pub_year,
                                      
                                      rec_pub_day_2 = rec_pub_day_2,
                                      rec_pub_year_2 = rec_pub_year_2,
                                      
                                      lag = lag,
                                      lag_2 = lag_2,
                                      
                                      cumu_cit_rec = cumu_cit_rec,
                                      cumu_cit_rec_2 = cumu_cit_rec_2,
                                      
                                      tfe = tfe,
                                      tfe_2 = tfe_2,
                                      
                                      sim = sim,
                                      sim_2 = sim_2)
        
        self.effects.export('effects/data_preprocessing.hdf5',progress=True)
        
