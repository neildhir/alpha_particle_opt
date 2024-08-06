import numpy as np
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition, proc0_print
from src.sample.particle_sampler import ParticleSampler
from src.trace.objectives_and_constraints import TraceBoozer, FieldStrength, prepare_config

vmec_input = "../vmec_input_files/nfp4/ours/input.nfp4_QH_cold_high_res_phase_one_mirror_1.35_aspect_7.0_iota_0.89"
max_mode = 3
aspect_target = 7.0
major_radius = 1.7 * aspect_target
target_volavgB = 1.0

mpi = MpiPartition(1)
vmec = Vmec(vmec_input, mpi=mpi, keep_all_files=False, verbose=False)
vmec = prepare_config(vmec, max_mode, major_radius, aspect_target, target_volavgB)
nfp = 4

# check the field strength call
fs = FieldStrength(vmec=vmec)
proc0_print(fs.compute())
proc0_print(fs.mirror_ratio())


# check particle tracing
sampler = ParticleSampler(nfp, s_label=0.25, n_particles = 20).sample_surface
tracer = TraceBoozer(
    vmec,
    mpi,
    sampler,
    tmax= 1e-1,
    tracing_tol= 1e-8,
    interpolant_degree=3,
    interpolant_level=8,
    bri_mpol= 8,
    bri_ntor = 8,
)

import time
proc0_print('tracing')
t0 = time.time()
c_times, is_success = tracer.compute_confinement_times()
t1 = time.time()
proc0_print(c_times)
proc0_print("time", t1 - t0)
energy_loss = tracer.energy_loss()
loss_fraction = tracer.loss_fraction()
proc0_print(energy_loss, loss_fraction)


