#!/usr/bin/env python

import os

import numpy as np

from simsopt.mhd import Vmec
from simsopt.objectives import ConstrainedProblem
from simsopt.solve import constrained_mpi_solve
from simsopt.util import MpiPartition, proc0_print

from src.sample.particle_sampler import NonUniformSampler
from src.trace.objectives_and_constraints import Booz, FastIonLoss, FieldStrength, prepare_config

"""
Solve the particle tracing problem.

min_w E_x[EnergyLoss(w)]
s.t. 
  B_lb <= B(x, w) <= B_ub

Run with e.g.
  mpiexec -n 48 optimization_example.py

(Any number of processors will work.)
"""

# initial configuration
vmec_input = "../vmec_input_files/nfp4/ours/input.nfp4_QH_cold_high_res"

# optimization variables
max_mode = 1
aspect_target = 7.0
major_radius = 1.7 * aspect_target
target_volavgB = 0.1
mirror_target = 1.35
n_particles = 30
s_label = 0.25
tmax=1e-4
tracing_tol=1e-8
interpolant_degree=3
interpolant_level=4
bri_mpol=4
bri_ntor=4

proc0_print("setting up problem")
proc0_print("==================================================")

mpi = MpiPartition(1)
vmec = Vmec(vmec_input, mpi=mpi, keep_all_files=False, verbose=False)
vmec = prepare_config(vmec, max_mode, major_radius, aspect_target, target_volavgB)
vmec.run()
nfp = vmec.wout.nfp

# (Optional) put bound constraints on the variables
# n_dofs = len(vmec.surf.x)
# vmec.surf.upper_bounds = 10*np.ones(n_dofs)
# vmec.surf.lower_bounds = -5*np.ones(n_dofs)
# surf.set_upper_bound("rc(1,0)", 1.0)

# boozer field
booz = Booz(vmec, interpolant_degree=interpolant_degree, interpolant_level=interpolant_level,
          bri_mpol=bri_mpol, bri_ntor=bri_ntor)

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

# constraint
fs = FieldStrength(vmec=vmec)
modB_lb = target_volavgB*2/(1+mirror_target)
modB_ub = target_volavgB*2*mirror_target/(1+mirror_target)
tuples_nlc = [(fs.modB, modB_lb, modB_ub)]

from simsopt.mhd import QuasisymmetryRatioResidual
# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|
#prob = ConstrainedProblem(qs.total, tuples_nlc=tuples_nlc)
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
    print('eval', 1, objective(x0+0.01))
    print('eval', 2, objective(x0))
    print('eval', 3, objective(x0))
    print('eval', 4, constraints[0].fun(x0))
    print('eval', 5, constraints[0].fun(x0+0.01))
    for c in constraints:
        print(4, c.fun(x0))

    # result must have result.x attribute
    result = type('Result', (), {})()
    result.x = x0
    return result


proc0_print("Running optimization")
proc0_print("==================================================")

loss = tracer.energy_loss()
proc0_print("Initial objective:", loss)
mirror = fs.mirror_ratio()
proc0_print("Initial mirror ratio:", mirror)

# solve the problem
constrained_mpi_solve(prob, mpi, opt_handle=bo_solver)
print('done')

# evaluate the solution
vmec.surf.x = prob.x
proc0_print("")
proc0_print(f"Completed optimization with max_mode ={max_mode}. ")
loss = tracer.energy_loss()
proc0_print("Final objective:", loss)
mirror = fs.mirror_ratio()
proc0_print("Final mirror ratio:", mirror)
