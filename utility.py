import pandas as pd
import numpy as np
import vaex as vx
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
#from joblib import Parallel, delayed
#from scipy.spatial.distance import cosine

import logging
logging.basicConfig(format='%(asctime)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

def device_identifier():
    try: #torch 2.0 currently under development (08/02/2023)
        if torch.has_mps:
            dev = 'mps'
    except:
        if torch.has_cuda:
            dev = 'cuda'
        else:
            dev = 'cpu'
    return dev


def remove_duplicates(df, grouping_cols: list):
        """Removes duplicated rows based on groupping columns and created index.

        1. Create index.
        2. Create new data frame, group on grouping columns and find minimum index
        3. Join on index and index_min
        4. Remove rows without index_min (duplicated)
        5. Returns deduplicated data frame
        """
        df["index"] = vx.vrange(0, df.shape[0])
        df_group = df.groupby(grouping_cols, agg=vx.agg.min("index"))
        df = df.join(df_group[["index_min"]], left_on="index", right_on="index_min")
        df = df[df.index_min.notna()]

        df = df.drop(["index", "index_min"])
        df = df.extract()
        
        return df


#def cos(row, sender_pos, receiver_pos):
#    a = sender_pos[row]
#    b = receiver_pos[row]
#    return 1-cosine(a, b)

def cos(sender_emb,receiver_emb):
    return np.sum(sender_emb*receiver_emb, axis=1)/(np.linalg.norm(sender_emb,axis=1)*np.linalg.norm(receiver_emb,axis=1))


def TexSim(sender_pos,rec_pos,emb_matrix,pub_date_y):
    
    #index = np.array(range(len(df)), dtype=np.int64)
    index = np.array([],dtype=np.int64)
    sim = np.array([])
    years = list(range(np.min(pub_date_y),
                       np.max(pub_date_y)+1))
    
    for y in tqdm(years):
        
        pos = np.where(pub_date_y == y)[0]
        #sender_pos_tmp = sender_pos[pos]
        #receiver_pos_tmp = rec_pos[pos]
        #index_tmp = index[pos]
        
        sender_emb = emb_matrix[sender_pos[pos]]
        rec_emb = emb_matrix[rec_pos[pos]]
        sim_curr = cos(sender_emb=sender_emb,
                       receiver_emb=rec_emb)
        
        #sim_curr =  Parallel(n_jobs=-1)(delayed(cos)(i, sender_emb, rec_emb) for i in range(len(pos)))
        
        sim = np.append(sim,sim_curr)
        index = np.append(index,pos)
        
    return vx.from_arrays(pos=index,sim=sim).sort('pos')

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

        
def ipc_classes(ipc:str,
                seq_len:int = 3):
    
    ipc = vx.from_pandas(pd.read_csv(ipc,low_memory=False)).drop('rank','version')
            
    ipc['ipc']= ipc['ipc'].str.slice(0,seq_len) 
    ipc = remove_duplicates(df=ipc,grouping_cols=['patnum','ipc'])
        
    logging.info('Mapping ipc classes')
    ipc_classes = ipc['ipc'].unique()
    ipc_dict = dict(zip(ipc_classes,range(len(ipc_classes))))
    tmp = ipc['ipc'].to_pandas_series()
    ipc_numeric = [ipc_dict[tmp[i]] for i in tqdm(range(len(tmp)))]
        
    ipc['ipc_numeric'] = np.asarray(ipc_numeric)
        
    logging.info('Grouping classes list per patent')
    ipc = ipc.to_pandas_df().groupby('patnum')['ipc_numeric'].apply(list)
    ipc = pd.DataFrame({'patnum':ipc.index,'classes':ipc})
        
    return vx.from_pandas(ipc)


