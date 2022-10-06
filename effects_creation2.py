import pandas as pd
import numpy as np
import vaex as vx
import pickle as pkl
from tqdm import tqdm

from joblib import Parallel, delayed
from scipy.spatial.distance import cosine
import os
import psutil
import gc


df = vx.open('big_n_general.hdf5')
df = df.sort('pub_date_days')
r_set = vx.open('risk_set.hdf5')
gc.collect()
print('Memory allocation in mb', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

################################
### PUBLICATION DATE IN DAYS ###
################################

pub_date_days = df['pub_date_days'].to_numpy()
pub_date_days_rec = df['pub_date_rec_days'].to_numpy()
pub_date_days_rec_2 = r_set['pub_date_rec_days'].to_numpy()

########################
### PUBLICATION YEAR ###
########################

pub_date = df['pub_date'].str.slice(0,4).to_numpy().astype('int')
pub_date_rec = df['pub_date_rec'].str.slice(0,4).to_numpy().astype('int')
pub_date_rec_2 = r_set['pub_date_rec'].str.slice(0,4).to_numpy().astype('int')

###########
### LAG ###
###########

lag_day = df['lag_days'].to_numpy()
lag_days_2 = df['pub_date_days'].to_numpy()-r_set['pub_date_rec_days'].to_numpy()

###############################################
### CUMULATIVE NUMBER OF CITATIONS RECEIVED ###
###############################################

cumu_cit_rec = df['cumu_cit_rec'].to_numpy()
cumu_cit_rec_2 = r_set['cumu_cit_rec'].to_numpy()

#################
### TRIANGLES ###
#################

tri = df['triangles_rec'].to_numpy()
tri_2 = r_set['triangles_rec'].to_numpy()

###################################
### TIME FROM PREVIOUS CITATION ###
###################################

tfe = df['time_from_prev_cit'].to_numpy()
tfe_2 = r_set['time_from_prev_cit'].to_numpy()


##################
### SIMILARITY ###
##################

sim = df['sim'].to_numpy()

with open('embeddings.pkl', "rb") as input_file:
    emb = pkl.load(input_file)['embeddings']

gc.collect()
print('Memory allocation in mb', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

def cos(row, sender_pos, receiver_pos):
    a = sender_pos[row]
    b = receiver_pos[row]
    return 1-cosine(a, b)

def TexSim(sender_pos,rec_pos,emb_matrix,pub_date_y):
    
    #index = np.array(range(len(df)), dtype=np.int64)
    index = np.array([],dtype=np.int64)
    sim = np.array([])
    years = list(range(np.min(pub_date_y),
                       np.max(pub_date_y)+1))
    
    for y in tqdm(years):
        
        pos = np.where(pub_date_y == y)[0]
        sender_pos_tmp = sender_pos[pos]
        receiver_pos_tmp = rec_pos[pos]
        #index_tmp = index[pos]
        
        sender_emb = emb_matrix[sender_pos_tmp]
        rec_emb = emb_matrix[receiver_pos_tmp]
        
        sim_curr =  Parallel(n_jobs=-1)(delayed(cos)(i, sender_emb, rec_emb) for i in range(len(pos)))
        
        sim = np.append(sim,sim_curr)
        index = np.append(index,pos)
    return sim, index

sim_2,_ = TexSim(sender_pos=df['sender_pos'].to_numpy().astype('int'),
                 rec_pos=r_set['receiver_pos'].to_numpy().astype('int'),
                 emb_matrix= emb,
                 pub_date_y=pub_date)

effects = vx.from_arrays(pub_date_days = pub_date_days,             # Event date of receiver in days from 01/01/1976
                         pub_date_days_rec = pub_date_days_rec,     # Publication date of receiver in days from 01/01/1976
                         pub_date_days_rec_2 = pub_date_days_rec_2, # Publication date of non-receiver in days from 01/01/1976
                         pub_date = pub_date,                       # Publication year of sender (event year)
                         pub_date_rec = pub_date_rec,               # Publication year of receiver
                         pub_date_rec_2 = pub_date_rec_2,           # Publication year of non-receiver
                         lag_days = lag_day,                        # Time lag of sender vs receiver 
                         lag_days_2 = lag_days_2,                   # Time lag of sender vs non-receiver
                         cumu_cit_rec = cumu_cit_rec,               # Cumulative citations received 
                         cumu_cit_rec_2 = cumu_cit_rec_2,           # Cumulative citations received of non-receiver
                         tri = tri,                                 # Triadic closure
                         tri_2 = tri_2,                             # Triadic closure non-receiver
                         sim = sim,                                 # Textual similarity of sender vs receiver
                         sim_2 = sim_2,                             # Textual similarity of sender vs non-receiver
                         tfe = tfe,                                 # Time from event
                         tfe_2 = tfe_2)                             # Time from event non-receiver 

effects.export('effects.hdf5',progress=True)