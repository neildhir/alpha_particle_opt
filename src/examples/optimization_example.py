#!/usr/bin/env python

import os

import numpy as np

from simsopt.mhd import Vmec
from simsopt.objectives import ConstrainedProblem
from simsopt.solve import constrained_mpi_solve
from simsopt.util import MpiPartition, proc0_print

from src.sample.particle_sampler import ParticleSampler
from src.trace.objectives_and_constraints import FastIonLoss, FieldStrength, prepare_config

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
mirror_target = 1.35
n_particles = 20
s_label = 0.25
tmax= 1e-1
tracing_tol= 1e-8
interpolant_degree=3
interpolant_level=8
bri_mpol= 8
bri_ntor = 8

proc0_print("setting up problem")
proc0_print("==================================================")

mpi = MpiPartition(1)
vmec = Vmec(vmec_input, mpi=mpi, keep_all_files=False, verbose=False)
vmec = prepare_config(vmec, max_mode, major_radius, aspect_target, target_volavgB)
vmec.run()
nfp = vmec.wout.nfp

# TODO: put bound constraints on the variables
# n_dofs = len(vmec.surf.x)
# vmec.surf.upper_bounds = 10*np.ones(n_dofs)
# vmec.surf.lower_bounds = -5*np.ones(n_dofs)
# surf.set_upper_bound("rc(1,0)", 1.0)

# initialize a particle sampler
sampler = ParticleSampler(nfp, s_label=s_label, n_particles = n_particles).sample_surface

# objective
tracer = FastIonLoss(
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

# constraint
fs = FieldStrength(vmec=vmec)
modB_lb = target_volavgB*2/(1+mirror_target)
modB_ub = target_volavgB*2*mirror_target/(1+mirror_target)
tuples_nlc = [(fs.modB, modB_lb, modB_ub)]

prob = ConstrainedProblem(tracer.energy_loss, tuples_nlc=tuples_nlc)

# set up the BO solver
def bo_solver(objective, x0, bounds, constraints, method, options):
    """
    Template class for the BO method. Must be of the form,
    result = bo_solver(objective, x0, bounds, constraints,
                 method, options) 
    where result.x returns the optimal point.

    objective: callable, function handle to the objective
    x0: array, incumbent solution
    bounds: list of tuples of lower and upper bounds, i.e. [(0.0, 1.0), ..., (-1.0, 4.0)]
    constraints: list containing any scipy.NonlinearConstraint and scipy.LinearConstraint instances.
    method: str.
    options: dict, dictionary of options.
    """
    print('Executing the BO loop')
    print(objective(x0))
    for c in constraints:
        print(c.fun(x0))

    # result must have result.x attribute
    result = type('Result', (), {})()
    result.x = x0
    return result


proc0_print("Running optimization")
proc0_print("==================================================")

proc0_print("Initial objective:", tracer.energy_loss())
proc0_print("Initial mirror ratio:", fs.mirror_ratio())

# solve the problem
constrained_mpi_solve(prob, mpi, opt_handle=bo_solver)

# evaluate the solution
vmec.surf.x = prob.x
proc0_print("")
proc0_print(f"Completed optimization with max_mode ={max_mode}. ")
proc0_print("objective:", tracer.energy_loss())
proc0_print("mirror ratio:", fs.mirror_ratio())
