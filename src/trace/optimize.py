#!/usr/bin/env python

import os

import numpy as np

from simsopt.mhd import Vmec
from simsopt.objectives import ConstrainedProblem
from simsopt.solve import constrained_mpi_solve
from simsopt.util import MpiPartition, proc0_print

from src.sample.particle_sampler import ParticleSampler
from src.trace.objectives_and_constraints import TraceBoozer, FieldStrength, prepare_config

"""
Optimize a VMEC equilibrium for quasi-helical symmetry (M=1, N=-1)
throughout the volume.

Solve as a constrained opt problem
min QH symmetry error
s.t. 
  aspect ratio <= 8
  -1.05 <= iota <= -1

Run with e.g.
  mpiexec -n 48 constrained_optimization.py

(Any number of processors will work.)
"""

# initial configuration
vmec_input = "../vmec_input_files/nfp4/ours/input.nfp4_QH_cold_high_res_phase_one_mirror_1.35_aspect_7.0_iota_0.89"

# optimization variables
max_mode = 3
aspect_target = 7.0
major_radius = 1.7 * aspect_target
target_volavgB = 1.0
n_particles = 20
s_label = 0.25
tmax= 1e-1
tracing_tol= 1e-8
interpolant_degree=3
interpolant_level=8
bri_mpol= 8
bri_ntor = 8


mpi = MpiPartition(1)
vmec = Vmec(vmec_input, mpi=mpi, keep_all_files=False, verbose=False)
vmec = prepare_config(vmec, max_mode, major_radius, aspect_target, target_volavgB)
vmec.run()
nfp = vmec.wout.nfp

proc0_print("Running optimization")
proc0_print("==================================================")


# TODO: put bound constraints on the variables
# surf = vmec.surf
# n_dofs = len(surf.x)
# surf.upper_bounds = 10*np.ones(n_dofs)
# surf.lower_bounds = -5*np.ones(n_dofs)
# surf.set_upper_bound("rc(1,0)", 1.0)
# vmec.surf = surf

# initialize a particle sampler
sampler = ParticleSampler(nfp, s_label=s_label, n_particles = n_particles).sample_surface

# objective
tracer = TraceBoozer(
    vmec,
    mpi,
    sampler,
    tmax=tmax,
    tracing_tol=tracing_tol,
    interpolant_degree=interpolant_degree,
    interpolant_level=interpolant_level,
    bri_mpol=bri_mpol,
    bri_ntor=bri_ntor,
)

# TODO: constraint bounds
fs = FieldStrength(vmec=vmec)
modB_lb = ...
modB_ub = ...
tuples_nlc = [(fs.compute, modB_lb, modB_ub)]


proc0_print("Initial objective:", tracer.energy_loss())
proc0_print("Initial mirror ratio:", fs.mirror_ratio())


prob = ConstrainedProblem(tracer.energy_loss, tuples_nlc=tuples_nlc)

# TODO: set up the BO solver
# solve the problem
constrained_mpi_solve(prob, mpi)

# evaluate the solution
vmec.surf.x = prob.x
proc0_print("")
proc0_print(f"Completed optimization with max_mode ={max_mode}. ")
proc0_print("objective:", tracer.energy_loss())
proc0_print("mirror ratio:", fs.mirror_ratio())
