#!/usr/bin/env python

import os

import numpy as np

from simsopt.mhd import Vmec
from simsopt.objectives import ConstrainedProblem
from simsopt.solve import constrained_mpi_solve
from simsopt.util import MpiPartition, proc0_print

from src.sample.particle_sampler import NonUniformSampler
from src.trace.objectives_and_constraints import Booz, FastIonLoss, FieldStrength, prepare_config
from src.bo.bo_solver import BoSolver


"""
Solve the particle tracing problem.

min_w E_x[EnergyLoss(w)]
s.t. 
  B_lb <= B(x, w) <= B_ub

Run with e.g.
  mpiexec -n 48 optimization_example.py

(Any number of processors will work.)
"""

# physics variables
max_mode = 3
aspect_target = 7.0
major_radius = 1.7 * aspect_target
target_volavgB = 1.0
mirror_target = 1.35
n_particles = 2
s_label = 0.25
tmax= 1e-4
tracing_tol= 1e-8
interpolant_degree=3
interpolant_level=8
bri_mpol= 8
bri_ntor = 8

# BO variables
max_iter = 10
num_restarts = 2
raw_samples = 32
method = 'trust-constr'
options={'maxiter':200}

vmec_input_files = []
directory = "../vmec_input_files/nfp4/ours"
for filename in os.listdir(directory):
    if filename.startswith("input.nfp4"):
        vmec_input_files.append(os.path.join(directory, filename))
# TODO: remove
# vmec_input_files = vmec_input_files[:2]
print(vmec_input_files)

proc0_print("setting up problem")
proc0_print("==================================================")

mpi = MpiPartition(1)
vmec = Vmec(vmec_input_files[0], mpi=mpi, keep_all_files=False, verbose=False)
vmec = prepare_config(vmec, max_mode, major_radius, aspect_target, target_volavgB)
vmec.run()

nfp = vmec.wout.nfp
dim_x = len(vmec.surf.x)

# boozer field
booz = Booz(vmec, bri_mpol=bri_mpol, bri_ntor=bri_ntor)

# initialize a particle sampler
sampler = NonUniformSampler(mpi = mpi, nfp = nfp, s_label=s_label, n_particles=n_particles).sample_surface

# objective
tracer = FastIonLoss(
    booz=booz,
    mpi=mpi,
    sampler=sampler,
    tmax=tmax,
    tracing_tol=tracing_tol,
)

# initialize train_X and train_Y
n_inputs = len(vmec_input_files)
train_X = np.zeros((n_inputs, dim_x))
train_Y = np.zeros((n_inputs, 1))
for ii, ff in enumerate(vmec_input_files):
    new_vmec = Vmec(ff, mpi=mpi, keep_all_files=False, verbose=False)
    new_vmec = prepare_config(new_vmec, max_mode, major_radius, aspect_target, target_volavgB)
    vmec.surf.x = new_vmec.surf.x
    el = tracer.energy_loss()
    train_X[ii] = vmec.surf.x
    train_Y[ii] = el

# bound constraints on the variables
factor = 1.0
vmec.surf.upper_bounds = np.max(train_X, axis=0)*factor
vmec.surf.lower_bounds = np.min(train_X, axis=0)*factor


# TODO: set up gradients of the field strength or switch to mean-cross sectional area constraint.
fs = FieldStrength(vmec=vmec)
modB_lb = target_volavgB*2/(1+mirror_target)
modB_ub = target_volavgB*2*mirror_target/(1+mirror_target)
# linear/nonlinear constraints
tuples_nlc = [(fs.modB, modB_lb, modB_ub)]

prob = ConstrainedProblem(tracer.energy_loss, tuples_nlc=tuples_nlc)

solver = BoSolver(train_X,
                train_Y,
                max_iter=max_iter,
                verbose=True,
                num_restarts=num_restarts,
                raw_samples=raw_samples
                )   

proc0_print("Running optimization")
proc0_print("==================================================")

loss = tracer.energy_loss()
proc0_print("Initial objective:", loss)
mirror = fs.mirror_ratio()
proc0_print("Initial mirror ratio:", mirror)

# solve the problem
constrained_mpi_solve(prob, mpi, grad=False,
                      opt_method = method,
                      options = options,
                      opt_handle=solver.solve)

# evaluate the solution
vmec.surf.x = prob.x
proc0_print("")
proc0_print(f"Completed optimization with max_mode ={max_mode}. ")
loss = tracer.energy_loss()
proc0_print("Initial objective:", loss)
mirror = fs.mirror_ratio()
proc0_print("Initial mirror ratio:", mirror)
