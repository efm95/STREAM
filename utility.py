import numpy as np
import vaex

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.spatial.distance import cosine


def remove_duplicates(df, grouping_cols: list):
        """Removes duplicated rows based on groupping columns and created index.

        1. Create index.
        2. Create new data frame, group on grouping columns and find minimum index
        3. Join on index and index_min
        4. Remove rows without index_min (duplicated)
        5. Returns deduplicated data frame
        """
        df["index"] = vaex.vrange(0, df.shape[0])
        df_group = df.groupby(grouping_cols, agg=vaex.agg.min("index"))
        df = df.join(df_group[["index_min"]], left_on="index", right_on="index_min")
        df = df[df.index_min.notna()]

        df = df.drop(["index", "index_min"])
        df = df.extract()
        
        return df


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