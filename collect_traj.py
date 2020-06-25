import numpy as np
from score import mdanatra
import glob

def make_dataset(rdata, n ,n_prev = 100):
           data, target =[], []
           maxlen = n
           if rdata.ndim == 1:
               for i in range(len(rdata)-maxlen):
                   data.append(rdata[i:i+maxlen])
           else:
               for m in range(len(rdata)):
                   for i in range(len(rdata[m])-maxlen):
                       data.append(rdata[m][i:i+maxlen])
           re_data = np.array(data).reshape(len(data),maxlen,1)
           return re_data

def min_max(x, axis=None):
        min = x.min(axis=axis, keepdims=True)
        max = x.max(axis=axis, keepdims=True)
        result = (x-min)/(max-min)
        return result

def collect():
    traj_list=[]
    traj_list_dir =glob.glob("protein/lyzo/ha/*/cyc*/cand*/can_CA.xtc", recursive=True)
    top_list_dir = glob.glob("protein/lyzo/ha/*/cyc*/cand*/CA.gro", recursive=True)
    mdtra = mdanatra()
    print(np.array(traj_list_dir).shape)
    for traj, top in zip(traj_list_dir, top_list_dir):
        mdtra.fitting(top,traj,"em_CA.gro")
        rmsd=mdtra.rmsd_()
        #PC = mdtra.PCA()
        #PC = PC.T[0]
        traj_list.append(rmsd)
    traj_list=np.array(traj_list)
    if traj_list.ndim != 3:
       traj_list = np.expand_dims(traj_list,axis=3)
    return traj_list
if __name__=="__main__": 
  a = collect()
  print(a.shape)
  np.save("data.npy",a)
  b = make_dataset(a, n=128)
  print(b.shape)
  c=min_max(b)
  print(c.shape)
