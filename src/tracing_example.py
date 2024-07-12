import os
import sys

sys.path.insert(0, os.getcwd())  # Manually add current directory (src in this case) to pythonpath

from src.trace.trace_boozer import TraceBoozer

import numpy as np
from mpi4py import MPI
from torch import tensor, Tensor, stack, zeros, ones, inf

# from sklearn.preprocessing import MinMaxScaler


# MPI stuff
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Choose an initial configuration
vmec_input_file = "/Users/z004mktz/Code/fusion/alpha_particle_opt/src/vmec_input_files/input.nfp4_QH_cold_high_res"  # Used the cold start (input.nfp4_QH_cold_high_res) of this file instead of the default warm start (input.nfp4_QH_warm_start_high_res)

# number of Fourier modes for optimization
max_mode = 1
d = 4 * max_mode**2 + 4 * max_mode

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

# Output constraints on mirror ratio

ns_B = 8  # maxB should be on boundary (so we could always just sample the boundary...)
ntheta_B = 16
nzeta_B = 16
smin = 0.02
smax = 1.0
len_B_field_out = ns_B * ntheta_B * nzeta_B
mirror_target = 1.35
eps_B = (mirror_target - 1.0) / (mirror_target + 1.0)
B_ub = target_volavgB * (1 + eps_B) * np.ones(len_B_field_out)  # upper bound, eq. 14
B_lb = target_volavgB * (1 - eps_B) * np.ones(len_B_field_out)  # lower bound, eq. 14

tracer = TraceBoozer(
    vmec_input_file,  # vmec input file
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
    ns_B=ns_B,
    ntheta_B=ntheta_B,
    nzeta_B=nzeta_B,
    smin=smin,
    smax=smax,
)

# sync seeds across MPI ranks
tracer.sync_seeds()

# get the optimization variables
x0 = tracer.x0


# Objective
def f(x):
    """
    objective for minimization:

    expected energy loss
      f = E[3.5*np.exp(-2*c_times/tmax)]

    x: array, vmec configuration variables
    """
    # sample particle positions (uniformly in theta,phi not in space)
    stz_inits, vpar_inits = tracer.sample_surface(n_particles, s_label)

    # ensure compatibility with C++ tracing
    stz_inits = np.ascontiguousarray(stz_inits)
    vpar_inits = np.ascontiguousarray(vpar_inits)

    # compute confinement times (heavy)
    c_times = tracer.compute_confinement_times(x, stz_inits, vpar_inits, tmax)

    if np.any(~np.isfinite(c_times)):
        # vmec failed here; return worst possible value
        c_times = np.zeros(len(vpar_inits))

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


def compute_B_field_vmec(x: np.ndarray) -> np.ndarray:
    """
    Use VMEC to compute the B field.
    """
    # Compute modB on a grid
    modB = tracer.compute_modB_vmec(x, ns=ns_B, ntheta=ntheta_B, nphi=nzeta_B, smin=smin, smax=smax)

    # VMEC failure
    # TODO: @misha check this failure condition. It is ambigous, I changed it from if modB == [] but that is not a good condition since it will try to compare (numerical) modB against an empty list which is undefined. Check line 391 of tracer_boozer.py if there is a better way to assign the empty array, can be we np.empty(vec_len) instead? I don't want to change it myself in case there are downstream effects that I don't know about.
    if not modB:
        return np.zeros(len_B_field_out)

    # print some stuff
    if rank == 0:
        print("B interval:", np.min(modB), np.max(modB))
        print("Mirror Ratio:", np.max(modB) / np.min(modB))

    return modB


def acqf_nonlinear_inequality_constraints() -> list[tuple[callable, bool]]:
    """
    This function returns the nonlinear inequality constraints for the acquisition function. Nonlinear inequality constraints: equation (13) and (14) of [1], section 4.2.

    References
    ----------
    [1] Bindel, David, Matt Landreman, and Misha Padidar. "Direct optimization of fast-ion confinement in stellarators." Plasma Physics and Controlled Fusion 65.6 (2023): 065012.
    """

    # XXX: we could have separate constraints per dimension

    def create_constraints():
        B_field_cache = {}

        def get_B_field(x: np.ndarray) -> dict[np.ndarray, np.ndarray]:
            """
            Functions caches the B field values for a given x so that we don't have to recompute it.

            Parameters
            ----------
            x : np.ndarray
                Input

            Returns
            -------
            dict[np.ndarray, np.ndarray]
                B field values for a given x
            """

            if x not in B_field_cache:
                B_field_cache[x] = compute_B_field_vmec(x)
            return B_field_cache[x]

        B_diff_lower = lambda x: -(B_lb - get_B_field(x))  # Negated to conform to optimize_acqf docstring instructions
        B_diff_upper = lambda x: -(get_B_field(x) - B_ub)  # Negated to conform to optimize_acqf docstring instructions

        return [(B_diff_lower, True), (B_diff_upper, True)]

    return create_constraints()


def build_BO_params() -> tuple[Tensor, Tensor, Tensor, list[tuple[callable, bool]]]:

    assert d == len(x0)  # Dimension of the input space (# of Fourier coefficients)

    # Features

    # Create the scaler object
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # Scale the values to the unit cube
    # scaled_values = scaler.fit_transform(x0.reshape(-1, 1))
    train_X = tensor(x0).view(1, -1)  # 1 x d

    # Targets

    # Standardise the output if vector valued
    # train_y0 = (y0 - y0.mean()) / y0.std()
    y0 = f(x0)
    train_Y = tensor([y0]).unsqueeze(-1)

    # Bounds and constraints

    # The upper and lower bounds and specified with inf because we have non-linear inequality constraints -- see docstring for optimize_acqf
    bounds = tensor([[-inf] * d, [inf] * d])
    constraints = acqf_nonlinear_inequality_constraints()

    return train_X, train_Y, bounds, constraints


def compute_B_field(x: np.ndarray) -> np.ndarray:

    raise DeprecationWarning("This function is deprecated. Use compute_B_field_vmec instead.")

    # Compute modB on a grid

    field, bri = tracer.compute_boozer_field(x)
    if field is None:
        return np.zeros(len_B_field_out)
    modB = tracer.compute_modB(field, bri, ns=ns_B, ntheta=ntheta_B, nphi=nzeta_B)
    if rank == 0:
        print("B interval:", np.min(modB), np.max(modB))
        print("Mirror Ratio:", np.max(modB) / np.min(modB))

    return modB


if __name__ == "__main__":
    # Test new B-field computation which only uses vmec
    out = compute_B_field_vmec(x0)
    print(out)
