#!/usr/bin/env python

import os
import numpy as np
from mpi4py import MPI
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve.mpi import least_squares_mpi_solve


"""
Optimize a VMEC equilibrium for quasi-helical symmetry (M=1, N=1)
throughout the volume.
"""

# This problem has 24 degrees of freedom, so we can use 24 + 1 = 25
# concurrent function evaluations for 1-sided finite difference
# gradients.
print("Running 2_Intermediate/QH_fixed_resolution.py")
print("=============================================")

mpi = MpiPartition()
#max_mode = 1
largest_mode = 4
input_config = "input.nfp4_QH_cold_high_res"
vmec_input = "../../vmec_input_files/" + input_config
outfile = input_config + "_quasysymmetry_opt"

vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
surf = vmec.boundary

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|

# Define objective function
prob = LeastSquaresProblem.from_tuples([(vmec.aspect, 7, 1),
                                        (qs.residuals, 0, 1)])

if mpi.proc0_world:
    print("Quasisymmetry objective before optimization:", qs.total())
    print("Total objective before optimization:", prob.objective())

#least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-8)

for step in range(largest_mode):
    max_mode = step + 1

    # VMEC's mpol & ntor will be 3, 4, 5, 6:
    vmec.indata.mpol = 3 + step
    vmec.indata.ntor = vmec.indata.mpol

    if mpi.proc0_world:
        print("")
        print("Beginning optimization with max_mode =", max_mode, \
              ", vmec mpol=ntor=", vmec.indata.mpol, \
              ". Previous vmec iteration = ", vmec.iter)

    # Define parameter space:
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode,
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)") # Major radius

    # Carry out the optimization for this step:
    least_squares_mpi_solve(prob, mpi, grad=True)

    if mpi.proc0_world:
        print("Done optimization with max_mode =", max_mode, \
              ". Final vmec iteration = ", vmec.iter)

        print("Final aspect ratio:", vmec.aspect())
        print("Quasisymmetry objective after optimization:", qs.total())
        print("Total objective after optimization:", prob.objective())
    
    # write the data to a file
    vmec.write_input(outfile + f"_max_mode_{max_mode}")

print("End of 2_Intermediate/QH_fixed_resolution.py")
print("============================================")
