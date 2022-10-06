import modin.pandas as pd 
import numpy as np 
import vaex as vx
#import pickle as pkl
from tqdm import tqdm
import gc

df = vx.open('big_n_general.hdf5')
df = df.sort('pub_date_days')# !!!!!!!!!! FONDAMENTALE !!!!!!!
#gc.collect()


unique_j = df['receiver','pub_date_rec','owner_receiver','receiver_pos','pub_date_rec_days','cumu_cit_rec','triangles_rec','time_from_prev_cit']
unique_j['index'] = vx.vrange(0,len(unique_j),1,dtype='int')

gc.collect()
#unique_j = remove_duplicates(unique_j,['receiver','pub_date_rec'])

day_count = df.groupby('pub_date_days',agg='count')
day_count = day_count.sort('pub_date_days').to_pandas_df()

#risk_set = vx.from_pandas(pd.DataFrame(columns=['receiver','pub_date_rec','owner_receiver','receiver_pos','pub_date_rec_days','cumu_cit_rec','triangles_rec']))

index = np.array([],dtype=np.int64)

for i in tqdm(range(len(day_count))):
    
    day = day_count.iloc[i,0]
    n_events = day_count.iloc[i,1]
    
    #Sampling non-events
    non_events = unique_j[unique_j['pub_date_rec_days']<day].index.to_numpy()
    #samp = non_events.sample(n_events,random_state=10091995+i,replace=True).to_numpy()
    np.random.RandomState(10091995+i)
    samp = np.random.choice(a=non_events,size=n_events,replace=True)
    index = np.append(index,samp)
    #gc.collect()
    
unique_j.drop('index',inplace=True)
unique_j = unique_j.to_pandas_df()
unique_j = unique_j.iloc[index,:]
unique_j = vx.from_pandas(unique_j)
unique_j.export('risk_set.hdf5',progress=True)

#index

#risk_set.export('risk_set_test.hdf5',progress=True)