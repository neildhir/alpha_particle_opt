import numpy as np
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition, proc0_print
from src.trace.objectives_and_constraints import FieldStrength, prepare_config
from src.utils.divide_work import divide_work
import os
from mpi4py import MPI
import pickle

"""
Find feasible points through random sampling in parallel.
1. randomly sample a bound constrained region.
2. evalute the field strength constraint.
3. only keep the feasible points.
"""

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_workers = comm.Get_size()

max_mode = 1
aspect_target = 7.0
major_radius = 1.7 * aspect_target
target_volavgB = 1.0
mirror_target = 1.35
n_samples = 2 # total samples we take (the number feasible will be less).

# input parameters for defining the feasible region.
vmec_input_files = ["../vmec_input_files/nfp4/ours/input.nfp4_QH_cold_high_res_phase_one_mirror_1.35_aspect_7.0_iota_1.043",
                    "../vmec_input_files/nfp4/ours/input.nfp4_QH_warm_start_high_res",
                    "../vmec_input_files/nfp4/ours/input.nfp4_QH_cold_high_res",
                    "../vmec_input_files/nfp4/ours/input.nfp4_QH_cold_high_res_mirror_feasible",
                    "../vmec_input_files/nfp4/ours/input.nfp4_QH_cold_high_res_phase_one_mirror_1.35_aspect_7.0_iota_0.89"
                    ]
print(vmec_input_files)

#==============================================================
# script starts here

# initialize vmec
mpi = MpiPartition()
vmec = Vmec(vmec_input_files[0], mpi=mpi, keep_all_files=False, verbose=False)
vmec = prepare_config(vmec, max_mode, major_radius, aspect_target, target_volavgB)
vmec.run()
nfp = vmec.wout.nfp

fs = FieldStrength(vmec=vmec)
modB_lb = target_volavgB*2/(1+mirror_target)
modB_ub = target_volavgB*2*mirror_target/(1+mirror_target)

def check_constraints(modB):
    """
    check the constraint
        modB <= ub
        modB >= lb
    rewrite them as 
        lb-modB <= 0
        modB - ub <= 0
    and take 
        c(x) = max(lb - modB, modB - ub)
    which should satisfy
        c(x) <= 0.

    return c(x)
    """
    c1 = np.max(modB - modB_ub)
    c2 = np.max(modB_lb - modB)
    return max(c1,c2)


# define the bound constraints
n_inputs = len(vmec_input_files)
dim_x = len(vmec.surf.x)
train_X = np.zeros((n_inputs, dim_x))

for ii, ff in enumerate(vmec_input_files):
    nvmec = Vmec(ff, mpi=mpi, keep_all_files=False, verbose=False)
    nvmec = prepare_config(nvmec, max_mode, major_radius, aspect_target, target_volavgB)
    train_X[ii] = np.copy(nvmec.surf.x)

factor = 1.0
sample_ub = np.max(train_X, axis=0)*factor
sample_lb = np.min(train_X, axis=0)*factor


# sample the points and share
sample_X = np.random.uniform(sample_lb, sample_ub, size=(n_samples, dim_x))
mpi.comm_groups.Bcast(sample_X, root=0)

# evaluate constraints
intervals, counts = divide_work(n_samples, n_workers)
n_points_local = counts[rank]
sample_Y_local = np.zeros(n_points_local)
sample_X_local = sample_X[intervals[rank]]
for ii, xx in enumerate(sample_X_local):
    vmec.surf.x = xx
    cx = check_constraints(fs.modB())
    sample_Y_local[ii] = cx

# gather evals
sample_Y = np.zeros(n_samples) # receive buffer
counts = np.array(counts).astype(int) # counts for each worker
mpi.comm_world.Gatherv(sample_Y_local, (sample_Y, counts), root=0)

# only keep the feasible ones
idx = sample_Y <= 0.0
sample_X = sample_X[idx]
sample_Y = sample_Y[idx]
if mpi.proc0_world:
    print('')
    print('feasibility fraction', np.mean(idx))

if mpi.proc0_world:
    # save it
    d = {}
    d['X'] = sample_X
    d['Y'] = sample_Y
    d['lb'] = sample_lb
    d['ub'] = sample_ub
    d["max_mode"] = max_mode
    d["aspect_target"] = aspect_target
    d["major_radius"] = major_radius
    d["target_volavgB"] = target_volavgB
    d["mirror_target"] = mirror_target
    pickle.dump(d, open("feasible_samples.pickle", "wb"))