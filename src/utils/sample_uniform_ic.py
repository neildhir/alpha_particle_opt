import numpy as np
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition, proc0_print
from src.trace.objectives_and_constraints import FieldStrength, prepare_config
import os

max_mode = 3
aspect_target = 7.0
major_radius = 1.7 * aspect_target
target_volavgB = 1.0
mirror_target = 1.35

# input parameters
vmec_input_files = []
directory = "../vmec_input_files/nfp4/ours"
for filename in os.listdir(directory):
    if filename.startswith("input.nfp4"):
        vmec_input_files.append(os.path.join(directory, filename))
print(vmec_input_files)


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


# storage
n_inputs = len(vmec_input_files)
dim_x = len(vmec.surf.x)
train_X = np.zeros((n_inputs, dim_x))
train_Y = np.zeros((n_inputs, 1))


for ii, ff in enumerate(vmec_input_files):
    nvmec = Vmec(ff, mpi=mpi, keep_all_files=False, verbose=False)
    nvmec = prepare_config(nvmec, max_mode, major_radius, aspect_target, target_volavgB)
    vmec.surf.x = nvmec.surf.x
    cx = check_constraints(fs.modB())

    train_X[ii] = np.copy(vmec.surf.x)
    train_Y[ii] = cx

print(train_Y)

