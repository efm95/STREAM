import numpy as np 
#import pandas as pd 
import torch
import torch.nn as nn 
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
gc.collect()

spline_df = [9,9,9,9,9]
model = REM_logit(p=sum(spline_df))
trainer = REM_sgd(model=model,lr=0.01)
trainer.fit(X=X,spline_df=spline_df)


print(f'AIC:{trainer.AIC}')
print(f'BIC: {trainer.BIC}')
coefs = model.linear.weight.detach().squeeze(0)

torch.save(coefs,'final_code_test.pt')
