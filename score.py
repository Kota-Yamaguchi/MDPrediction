import MDAnalysis
import sys
import numpy as np
from MDAnalysis.core.universe import Universe
from MDAnalysis import transformations
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from pandas import Series 
import mdtraj as md
from MDAnalysis.analysis.rms import rmsd
from MDAnalysis.analysis import align

class mdanatra():
    def __init__(self):
        self.reference = None
        self.trajectory = None
        self.trj = None
        self.frame = None
        self.atom = None
        self.ref = None
#sys[2] is xtc
#sys[1] is gro
#sys[3] is reference structure
#below code is fitting.

    def __repr__(self):
        return "Frame,Atom : [{}, {}]".format(self.frame, self.atom)

    def fitting(self, top, xtc, ref):
        trj = Universe(top, xtc)
        ref = Universe(ref)
        self.ref = ref
        tran1=transformations.fit_rot_trans(trj, ref, plane="xy", weights="mass")
        tran2=transformations.fit_rot_trans(trj, ref, plane="xz", weights="mass")
        tran = [tran1, tran2]
        trj.trajectory.add_transformations(*tran)
        self.trj = trj
#object structure
        d = map(lambda i : np.array(i), trj.trajectory)
        d = np.array(list(d))
        m = map(lambda i : d[i].reshape(492), range(d.shape[0]))
        self.trajectory = np.array(list(m))
        self.frame = len(trj.trajectory)
        self.atom = trj.atoms
#reference structure
        ref = map(lambda i : np.array(i), ref.trajectory)
        ref = np.array(list(ref))
        ref_vec = map(lambda i : ref[i].reshape(492), range(ref.shape[0]))
        self.reference = np.array(list(ref_vec))

    def mean_structure(self):
        average = np.mean(self.trajectory)
        self.reference = average

    def rmsd_(self):        
        #print("calculate RMSD ...")
        rmsd_ = np.array([])
        for ts in self.trj.trajectory:
           a =rmsd(self.trj.select_atoms("name CA").positions, self.ref.select_atoms("name CA").positions,
         superposition = True)
           rmsd_ = np.append(rmsd_, a)
        #rmsd = np.sqrt(np.sum((self.trajectory - self.reference)**2,axis = 1)) 
        #rmsd = np.sqrt(mean_squared_error(self.trajectory[0], self.reference[0]))
        #np.save("rmsd.npy",rmsd)
        print(rmsd_.shape)
        return rmsd_


    def ranking(self, rmsd, rank = 10, reverse = True):
        hist = rmsd
        rank_argv = []
        top_rank = sorted(hist, reverse = reverse)[:int(rank)]
        for n in range(len(Series(top_rank))):
            for i in range(len(hist)):
                if hist[i] == Series(top_rank)[n]:
                    rank_argv.append(i) 
        score = np.array([rank_argv, top_rank])
        return score

    def domain_distance(self, res_number_C, res_number_N):
#        if type(res_number_C) or type(res_number_N) != list:
#            raise Exception("input list format, not int and float")
        C_first = res_number_C[0]
        C_last = res_number_C[1]
        N_first = res_number_N[0]
        N_last = res_number_N[1] 
        n_ter = self.trj.select_atoms("protein and resid {}:{}".format(N_first, N_last))
        c_ter = self.trj.select_atoms("protein and resid {}:{}".format(C_first, C_last))
        n_ter_g = np.array([])
        c_ter_g = np.array([])
        for ts in self.trj.trajectory:
            n_ter_g=np.append(n_ter_g,n_ter.select_atoms("protein").center_of_mass())
            c_ter_g=np.append(c_ter_g,c_ter.select_atoms("protein").center_of_mass())
    
        n_ter_g = n_ter_g.reshape(len(self.trj.trajectory),3)
        c_ter_g = c_ter_g.reshape(len(self.trj.trajectory),3)

        OC = c_ter_g - n_ter_g
        OC = np.linalg.norm(OC, axis=1)
        return OC
    
    def PCA(self):
        pca = PCA(n_components=3)
        pca.fit(self.trajectory)
        projection = pca.transform(self.trajectory)
        return projection

    def translater(self, top, xtc, frame, number):
        traj = md.load_xtc(xtc, top=top)
        traj = traj[frame]
        traj.save_gro("candi{0}_{1}.gro".format(number,frame))

    
if __name__ == "__main__":
    mdtra = mdanatra()
    mdtra.fitting(sys.argv[1],sys.argv[2],sys.argv[3])
    rmsd = mdtra.rmsd()
    score_rmsd = mdtra.ranking(rmsd, rank=10, reverse = True)
    print(score_rmsd)
    mdtra.transfer
    OC = mdtra.domain_distance([74,162],[1,65])
    score_OC = mdtra.ranking(OC, rank = 10, reverse = False)
    print(score_OC)
