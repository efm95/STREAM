import pandas as pd
import numpy as np
import vaex as vx
import gc

import sys

sys.path.insert(1,'')

from utility import *

##########################
### NODE LIST CLEANING ###
##########################

class node_list_clean:
    
    def __init__(self,
                 grant: str,
                 ipc: str):
        
        self.grant = vx.from_pandas(pd.read_csv(grant,low_memory=False))
        
        ipc = vx.from_pandas(pd.read_csv(ipc,low_memory=False))
        self.ipc = ipc.groupby(['patnum'],agg='count')
                
    def fit(self):
        #Removing potential duplicates duplicates
        self.grant = remove_duplicates(df = self.grant,grouping_cols = ['patnum'])
                
        #Combine grant with ipc count
        self.grant = self.grant.join(self.ipc,on='patnum',how='inner',allow_duplication=False)
        self.grant.rename('count','ipc_count')
        gc.collect()
        
        #Removing utility patents --> patents with no-abstract
        self.grant = self.grant[self.grant['abstract'].ismissing()==False]
        gc.collect()
        
        #Save abstracts for text similarity computation
        abstracts = self.grant['patnum','abstract']
        abstracts.export('data_preprocessing/abstracts.hdf5',progress=True)
        
        #Save cleaned node list
        grant_reduced = self.grant['patnum','pubdate','ipc_count']
        
        if grant_reduced['pubdate'].dtype != str:
            grant_reduced['pubdate'] = grant_reduced['pubdate'].astype('int').astype('str')
        
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

        self.cit.export('data_preprocessing/edge_list.hdf5',progress=True)
