import pandas as pd
import numpy as np
import vaex as vx
import gc

import logging
logging.basicConfig(format='%(asctime)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

#import sys

#sys.path.insert(1,'')

from utility import *

##########################
### NODE LIST CLEANING ###
##########################

class node_list_clean:    
    def __init__(self,
                 grant: str):
        
        self.grant = vx.from_pandas(pd.read_csv(grant,low_memory=False))
                
    def fit(self):
        
        #Removing potential duplicates duplicates
        logging.info('Removing duplicates in grant')
        self.grant = remove_duplicates(df = self.grant,grouping_cols = ['patnum'])
        
        logging.info('removing missing values from astract list')
        #Removing utility patents --> patents with no-abstract
        self.grant = self.grant[self.grant['abstract'].ismissing()==False]
        gc.collect()
        
        #Save abstracts for text similarity computation
        logging.info('Saving abstract list')
        abstracts = self.grant['patnum','abstract']
        abstracts.export('data_preprocessing/abstracts.hdf5',progress=True)
        
        #Save cleaned node list
        grant_reduced = self.grant['patnum','pubdate']
        #self.grant['embedding_pos'] =  np.array(range(len(self.grant)))
        
        if grant_reduced['pubdate'].dtype != str:
            grant_reduced['pubdate'] = grant_reduced['pubdate'].astype('int')#.astype('str')
        
        grant_reduced = grant_reduced.to_pandas_df()
        
        grant_reduced['pub_year'] = grant_reduced['pubdate'].astype('str').str.slice(0,4).astype('int')
        pub_date = grant_reduced['pubdate']#.to_pandas_series()
        pub_date = pd.to_datetime(pub_date,format="%Y%m%d")
        starting_day = pd.Timestamp('1976-01-01')
        
        grant_reduced['pub_date_days'] = (pub_date-starting_day).dt.days.astype('int')
        grant_reduced.drop(['pubdate'],inplace=True,axis=1)
        
        grant_reduced = vx.from_pandas(grant_reduced)
        grant_reduced['embedding_pos'] = vx.vrange(0,len(grant_reduced),dtype='int')
        logging.info('Saving reduced node list in hdf5 format')
        grant_reduced.export('data_preprocessing/grant_reduced.hdf5',progress=True)
        


#################
### EDGE LIST ###
#################
   
class edge_list_clean: 
    def __init__(self,
                 cit:str):
        self.cit = vx.from_pandas(pd.read_csv(cit,low_memory=False))
        
    def fit(self):
        self.cit.rename('src','sender')
        self.cit.rename('dst','receiver')
        
        logging.info('Saving edge_list in hdf5 format')
        self.cit.export('data_preprocessing/edge_list.hdf5',progress=True)

        
