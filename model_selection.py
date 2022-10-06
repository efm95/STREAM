import numpy as np 
import torch
import vaex as vx

from bspline import * 
from rem_lkl import *

import gc 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eff_original = vx.open('effects.hdf5')
eff = eff_original.sample(frac=1,random_state=10091995)
gc.collect()

pub_date = torch.tensor(eff.pub_date_rec.to_numpy(),dtype=torch.float32)
pub_date2 = torch.tensor(eff.pub_date_rec_2.to_numpy(),dtype=torch.float32)

sim = torch.tensor(eff.sim.to_numpy(),dtype=torch.float32)
sim2 = torch.tensor(eff.sim_2.to_numpy(),dtype=torch.float32)

lag = torch.tensor(eff.lag_day.to_numpy(),dtype=torch.float32)
lag2 = torch.tensor(eff.lag_days_2.to_numpy(),dtype=torch.float32)

cumu_cit = torch.tensor(np.log(eff.cumu_cit_rec.to_numpy()),dtype=torch.float32)
cumu_cit_2 = torch.tensor(np.log(eff.cumu_cit_rec_2.to_numpy()),dtype=torch.float32)

tri = torch.tensor(eff.tri.to_numpy(),dtype=torch.float32)
tri_2 = torch.tensor(eff.tri_2.to_numpy(),dtype=torch.float32)


X = torch.column_stack((pub_date,pub_date2,sim,sim2,lag,lag2,cumu_cit,cumu_cit_2,tri,tri_2))
del pub_date, pub_date2, sim, sim2, lag, lag2, cumu_cit, cumu_cit_2, tri, tri_2
gc.collect()

index_long = torch.tensor(range(len(X)))
index = index_long.split(int(len(index_long)/5))

print(f'Number of folds: {len(index)}')

df_grid = list(range(4,17,1))
test_loss_mat = torch.zeros(len(index),len(df_grid))

for d in range(len(df_grid)):
    for rep in range(len(index)):
        
        index_test = index[rep]
        mask = torch.ones((len(index_long)),dtype=torch.bool)
        mask[index_test] = False
        
        X_test = X[index_test,:]
        X_train = X[mask,:]
        
        spline_df = [df_grid[d]]*int(X.shape[1]/2)
        trainer = REM_sgd(spline_df=spline_df,lr=0.01)
        trainer.fit(X=X_train)
        test_loss = trainer.test_loss(X_test)
        
        test_loss_mat[rep,d] = test_loss
        
    
torch.save(test_loss_mat,'model_selection.pt')