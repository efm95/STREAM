import pandas as pd
import numpy as np
import vaex as vx
import logging
import pickle
from sentence_transformers import SentenceTransformer, LoggingHandler

import timeit

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
        
    
class SBERT:
    def __init__(self):
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model.max_seq_length = 256
        
    def fit(self):

        start = timeit.default_timer()
        df = vx.open('data_preprocessing/abstracts.hdf5').to_pandas_df()
        
        #Start the multi-process pool on all available CUDA devices
        pool = self.model.start_multi_process_pool()
        
        #Compute the embeddings using the multi-process pool
        emb = self.model.encode_multi_process(df['abstract'], pool)
        self.model.stop_multi_process_pool(pool)
        
        print("Embeddings computed. Shape:", emb.shape)
            
        #Store embeddings
        with open('data_preprocessing/embeddings.pkl', "wb") as fOut:
            pickle.dump({'embeddings': emb}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    
        stop = timeit.default_timer()
        print('Time: ', stop - start) 