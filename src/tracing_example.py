import os
import sys

sys.path.insert(0, os.getcwd())

from src.trace.trace_boozer import TraceBoozer

import numpy as np
from mpi4py import MPI


# MPI stuff
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# choose an initial configuration
vmec_input = "../vmec_input_files/input.nfp4_QH_warm_start_high_res"

# number of Fourier modes for optimization
max_mode = 1

# target aspect ratio (ARIES-CS)
aspect_target = 7.0

# fixed major radius (ARIES-CS size)
major_radius = 1.7 * aspect_target

# volume average field strength
target_volavgB = 1.0  # tesla

# tracing parameters
s_label = 0.25  # surface label
tmax = 1e-4  # max tracing time
n_particles = 100  # number of particles

# tracing fidelity
tracing_tol = 1e-8
interpolant_degree = 3
interpolant_level = 8
bri_mpol = 8
bri_ntor = 8

tracer = TraceBoozer(
    vmec_input,  # vmec input file
    n_partitions=1,  # number of partitions for vmec (always use 1)
    max_mode=max_mode,  # maximum fourier modes for boundary
    major_radius=major_radius,  # major radius
    aspect_target=aspect_target,  # aspect ratio
    target_volavgB=target_volavgB,
    tracing_tol=tracing_tol,
    interpolant_degree=interpolant_degree,
    interpolant_level=interpolant_level,
    bri_mpol=bri_mpol,
    bri_ntor=bri_ntor,
)
# sync seeds across MPI ranks
tracer.sync_seeds()

# get the optimization variables
x0 = tracer.x0


def objective(x):
    """
    objective for minimization:

    expected energy retatined
      f = E[3.5*np.exp(-2*c_times/tmax)]

    x: array,vmec configuration variables
    """
    # sample particle positions (uniformly in theta,phi not in space)
    stz_inits, vpar_inits = tracer.sample_surface(n_particles, s_label)

    # ensure compatibility with C++ tracing
    stz_inits = np.ascontiguousarray(stz_inits)
    vpar_inits = np.ascontiguousarray(vpar_inits)

    # compute confinement times
    c_times = tracer.compute_confinement_times(x, stz_inits, vpar_inits, tmax)

    if np.any(~np.isfinite(c_times)):
        # vmec failed here; return worst possible value
        c_times = np.zeros(len(vpars))

    # energy retained by particle
    feat = 3.5 * np.exp(-2 * c_times / tmax)

    # sample average
    res = np.mean(feat)
    loss_frac = np.mean(c_times < tmax)

    # print with MPI
    if rank == 0:
        print("obj:", res, "P(loss):", loss_frac)
    sys.stdout.flush()

    return res


# TODO: now call a BO routine on rank 0
