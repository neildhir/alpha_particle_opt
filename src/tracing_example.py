import os
import sys

sys.path.insert(0, os.getcwd())  # Manually add current directory (src in this case) to pythonpath

from src.trace.trace_boozer import TraceBoozer

import numpy as np
from mpi4py import MPI
from torch import tensor, Tensor, stack, zeros, ones
from sklearn.preprocessing import MinMaxScaler


# MPI stuff
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# choose an initial configuration
# vmec_input = "../vmec_input_files/vmec_input_files/input.nfp4_QH_warm_start_high_res"
vmec_input = "/Users/z004mktz/Code/fusion/alpha_particle_opt/src/vmec_input_files/input.nfp4_QH_warm_start_high_res"

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


# Objective
def f(x):
    """
    objective for minimization:

    expected energy retatined
      f = E[3.5*np.exp(-2*c_times/tmax)]

    x: array, vmec configuration variables
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
        c_times = np.zeros(len(vpars))  # TODO: Misha to fix this; vpars is undefined

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


def build_train_data() -> tuple[Tensor, Tensor, Tensor]:

    d = len(x0)  # Dimension of the input space (# of Fourier coefficients)
    # Create the scaler object
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # Scale the values to the unit cube
    # scaled_values = scaler.fit_transform(x0.reshape(-1, 1))
    train_X = tensor(x0).view(1, -1)  # 1 x d

    # Standardise the output if vector valued
    # train_y0 = (y0 - y0.mean()) / y0.std()
    y0 = f(x0)
    train_Y = tensor([y0]).unsqueeze(-1)
    bounds = stack([zeros(d), ones(d)])

    return train_X, train_Y, bounds
