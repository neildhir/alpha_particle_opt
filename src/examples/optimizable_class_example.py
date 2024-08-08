import numpy as np
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition, proc0_print
from src.sample.particle_sampler import NonUniformSampler, UniformSampler
from src.trace.objectives_and_constraints import Booz, FastIonLoss, FieldStrength, prepare_config

# input parameters
vmec_input = "../vmec_input_files/nfp4/ours/input.nfp4_QH_cold_high_res_phase_one_mirror_1.35_aspect_7.0_iota_0.89"
max_mode = 3
aspect_target = 7.0
major_radius = 1.7 * aspect_target
target_volavgB = 1.0
tmax = 1e-4
tracing_tol = 1e-4
bri_mpol = 8
bri_ntor = 8
n_particles = 4
s_label = 0.25

mpi = MpiPartition(1)

# initialize vmec
vmec = Vmec(vmec_input, mpi=mpi, keep_all_files=False, verbose=False)
vmec = prepare_config(vmec, max_mode, major_radius, aspect_target, target_volavgB)
vmec.run()
nfp = vmec.wout.nfp

# check the field strength call
fs = FieldStrength(vmec=vmec)
modB = fs.modB()
mirror_ratio = fs.mirror_ratio()
proc0_print("")
proc0_print("field strength")
proc0_print(modB)
proc0_print(mirror_ratio)


# boozer field
booz = Booz(vmec, bri_mpol=bri_mpol, bri_ntor=bri_ntor)

# check the sampler
sampler = NonUniformSampler(mpi = mpi, nfp = nfp, s_label=s_label, n_particles=n_particles)
surf_samples = sampler.sample_surface()
v_samples = sampler.sample_volume()
proc0_print("")
proc0_print("Nonuniform sampler")
proc0_print(surf_samples)
proc0_print(v_samples)

sampler = UniformSampler(booz=booz, mpi=mpi, nfp=nfp, s_label=s_label, n_particles=n_particles)
surf_samples = sampler.sample_surface()
v_samples = sampler.sample_volume()
proc0_print("")
proc0_print("Uniform sampler")
proc0_print(surf_samples)
proc0_print(v_samples)

# check particle tracing
sampler = NonUniformSampler(mpi = mpi, nfp = nfp, s_label=s_label, n_particles=n_particles).sample_surface
tracer = FastIonLoss(
    booz=booz,
    mpi=mpi,
    sampler=sampler,
    tmax=tmax,
    tracing_tol=tracing_tol,
)

import time
proc0_print('tracing')
t0 = time.time()
c_times, is_success = tracer.compute_confinement_times()
t1 = time.time()
energy_loss = tracer.energy_loss()
loss_fraction = tracer.loss_fraction()
mean_ctime = tracer.mean_confinement_time()
proc0_print(c_times)
proc0_print("time", t1 - t0)
proc0_print(energy_loss, loss_fraction, mean_ctime)


