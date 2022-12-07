import numpy as np
from mpi4py import MPI
import sys
import pickle
from pdfo import pdfo,NonlinearConstraint as pdfo_nlc
#from skquant.opt import minimize as skq_minimize
from scipy.optimize import differential_evolution, NonlinearConstraint as sp_nlc, minimize as sp_minimize
from scipy.integrate import simpson
debug = False
if debug:
  sys.path.append("../../utils")
  sys.path.append("../../trace")
  sys.path.append("../../sample")
  #sys.path.append("../../opt")
  #sys.path.append("../../../SIMPLE/build/")
else:
  sys.path.append("../../../utils")
  sys.path.append("../../../trace")
  sys.path.append("../../../sample")
  #sys.path.append("../../../opt")
  #sys.path.append("../../../../SIMPLE/build/")
#from trace_simple import TraceSimple
from trace_boozer import TraceBoozer
from eval_wrapper import EvalWrapper
from radial_density import RadialDensity
from constants import V_MAX
from gauss_quadrature import gauss_quadrature_nodes_coeffs
#from sid_psm import SIDPSM

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

"""
Optimize a configuration to minimize alpha particle losses

ex.
  mpiexec -n 1 python3 optimize.py grid 0.5 mean_energy bobyqa 1 nfp4_phase_one None 0.0001 -1.043 7.0 5 5 5 5
"""


# configuration parmaeters
n_partitions = 1
#aspect_target = 8.0
#iota_target = 0.42 # only for QA
minor_radius = 1.7
#major_radius = 13.6
target_volavgB = 5.0
s_min = 0.0
s_max = 1.0
# optimizer params
maxfev = 500
max_step = 1.0 # for max_mode=1
#max_step = 0.1 # for max_mode=2
#max_step = 5e-2 # for max_mode=3
#max_step = 1e-3 # for max_mode=4
min_step = 1e-8
# trace boozer params
tracing_tol = 1e-8
interpolant_degree = 3
interpolant_level  = 8
bri_mpol = 16
bri_ntor = 16


# read inputs
sampling_type = sys.argv[1] # random or grid or SAA
sampling_level = sys.argv[2] # "full" or a float surface label
objective_type = sys.argv[3] # mean_energy or mean_time
method = sys.argv[4] # optimization method
max_mode = int(sys.argv[5]) # max mode
vmec_label = sys.argv[6] # vmec file
warm_start_file = sys.argv[7] # filename or "None"
tmax = float(sys.argv[8]) # tmax
constrain_iota = (sys.argv[9] != "None") # None or float
if constrain_iota:
  iota_target = float(sys.argv[9])
else:
  iota_target = 0.0
aspect_target = float(sys.argv[10]) # float
ns = int(sys.argv[11])  # number of surface samples
ntheta = int(sys.argv[12]) # num theta samples
nzeta = int(sys.argv[13]) # num phi samples
nvpar = int(sys.argv[14]) # num vpar samples
assert sampling_type in ['random', "grid", "SAA"]
assert objective_type in ['mean_energy','mean_time'], "invalid objective type"
assert method in ['cobyla','bobyqa','snobfit','diff_evol','nelder','sidpsm'], "invalid optimiztaion method"

# set the major radius
major_radius = minor_radius*aspect_target

n_particles = ns*ntheta*nzeta*nvpar
if objective_type == "mean_energy":
  ftarget = 3.5*np.exp(-2)
elif objective_type == "mean_time":
  ftarget = 0.0


if vmec_label == "nfp2_QA_cold_high_res":
  vmec_input="../../vmec_input_files/input.nfp2_QA_cold_high_res"
elif vmec_label == "nfp2_QA_cold_high_res_mirror_feas":
  vmec_input="../../vmec_input_files/input.nfp2_QA_cold_high_res_mirror_feasible"
elif vmec_label == "nfp2_QA_high_res":
  vmec_input="../../vmec_input_files/input.nfp2_QA_high_res"
elif vmec_label == "nfp4_QH_warm_high_res":
  vmec_input="../../vmec_input_files/input.nfp4_QH_warm_start_high_res"
elif vmec_label == "nfp4_QH_cold_high_res":
  vmec_input="../../vmec_input_files/input.nfp4_QH_cold_high_res"
elif vmec_label == "nfp4_QH_cold_high_res_mirror_feas":
  vmec_input="../../vmec_input_files/input.nfp4_QH_cold_high_res_mirror_feasible"
elif vmec_label == "nfp4_phase_one":
  vmec_input="../phase_one/input.nfp4_QH_cold_high_res_phase_one"
elif vmec_label == "nfp2_phase_one":
  vmec_input="../phase_one/input.nfp2_QA_cold_high_res_phase_one"

if not debug:
  vmec_input="../" + vmec_input


# load a starting point
if warm_start_file != "None": 
  data_x0 = pickle.load(open(warm_start_file,"rb"))
  x0 = data_x0['xopt']
  x0_max_mode = data_x0['max_mode']
  del data_x0
else:
  x0 = []
  x0_max_mode=max_mode

# build a tracer object
#tracer = TraceSimple(vmec_input,n_partitions=n_partitions,max_mode=max_mode,major_radius=major_radius)
tracer = TraceBoozer(vmec_input,
                      n_partitions=n_partitions,
                      max_mode=max_mode,
                      major_radius=major_radius,
                      aspect_target=aspect_target,
                      target_volavgB=target_volavgB,
                      tracing_tol=tracing_tol,
                      interpolant_degree=interpolant_degree,
                      interpolant_level=interpolant_level,
                      bri_mpol=bri_mpol,
                      bri_ntor=bri_ntor,
                      x0=x0,
                      x0_max_mode=x0_max_mode)
tracer.sync_seeds()
x0 = tracer.x0
dim_x = tracer.dim_x


# aspect constraint
def aspect_ratio(x):
  """
  Compute the aspect ratio
  """
  # update the surface
  tracer.surf.x = np.copy(x)

  # evaluate the objectives
  try:
    asp = tracer.surf.aspect_ratio()
  except:
    asp = np.inf

  # catch partial failures
  if np.isnan(asp):
    asp = np.inf

  if rank == 0:
    print("aspect",asp)
  return asp

def rotational_transform(x):
  """
  Compute iota
  """
  # update the surface
  tracer.surf.x = np.copy(x)

  # evaluate the objectives
  try:
    iota = tracer.vmec.mean_iota()
  except:
    iota = np.inf

  # catch partial failures
  if np.isnan(iota):
    iota = np.inf

  if rank == 0:
    print("iota",iota)
  return iota


# TODO: switch to a constraint that is free of boozxform
# constraint on mirror ratio
ns_B=8 # maxB should be on boundary
ntheta_B=nzeta_B=16
len_B_field_out = ns_B*ntheta_B*nzeta_B
def B_field(x):
  """
  Compute modB on a grid
  """
  field,bri = tracer.compute_boozer_field(x)
  if field is None:
    return np.zeros(len_B_field_out)
  modB = tracer.compute_modB(field,bri,ns=ns_B,ntheta=ntheta_B,nphi=nzeta_B)
  if rank == 0:
    print("B interval:",np.min(modB),np.max(modB))
    print("Mirror Ratio:",np.max(modB)/np.min(modB))
  return modB
B_mean = 5.0
mirror_target = 1.35
eps_B = (mirror_target - 1.0)/(mirror_target + 1.0)
B_ub = B_mean*(1 + eps_B)*np.ones(len_B_field_out)
B_lb = B_mean*(1 - eps_B)*np.ones(len_B_field_out)

#mirror_target = 1.35
#eps_B = (mirror_target - 1.0)/(mirror_target + 1.0)
#def B_field_volavg_con(x):
#  """
#  Use volavgB instead of 5 as the middle point of
#  the B field constraints:
#    volavgB(1-eps) <= B <= volavgB(1+eps)
#  where 
#    eps = (1.35 -1)/(1.35 + 1).
#
#  return inequality constraints c(x) <= 0
#    B - volavgB(1+eps) <= 0
#    volavgB(1-eps) - B <= 0
#  """
#  field,bri = tracer.compute_boozer_field(x)
#  if field is None:
#    return np.zeros(2*len_B_field_out)
#  modB = tracer.compute_modB(field,bri,ns=ns_B,ntheta=ntheta_B,nphi=nzeta_B)
#  if rank == 0:
#    print("B interval:",np.min(modB),np.max(modB))
#    print("Mirror Ratio:",np.max(modB)/np.min(modB))
#  # get volavgB
#  B_mean = tracer.vmec.wout.volavgB
#  rhs = B_mean*(1 + eps_B)
#  lhs = B_mean*(1 - eps_B)
#  # constraints c(x) <= 0
#  c_ub = modB - rhs
#  c_lb = lhs - modB
#  con = np.append(c_ub,c_lb)
#  return np.copy(con)
#B_ub = np.zeros(2*len_B_field_out)
#B_lb = -np.inf*np.ones(2*len_B_field_out)
  

"""
Generate points for tracing
"""

def get_random_points(sampling_level):
  """
  Return a set of random points to trace.
  """
  # sync seeds
  tracer.sync_seeds()
  if sampling_level == "full":
    # volume sampling
    stz_inits,vpar_inits = tracer.sample_volume(n_particles)
  else:
    # surface sampling
    s_label = float(sampling_level)
    stz_inits,vpar_inits = tracer.sample_surface(n_particles,s_label)
  return stz_inits,vpar_inits

# make a sampler for computing probabilities
radial_sampler = RadialDensity(1000)

if sampling_type == "grid" and sampling_level == "full":
  # simpson
  #s_lin = np.linspace(s_min,s_max, ns)
  #theta_lin = np.linspace(0, 2*np.pi, ntheta)
  #zeta_lin = np.linspace(0,2*np.pi/tracer.surf.nfp, nzeta)
  #vpar_lin = np.linspace(-V_MAX,V_MAX,nvpar)
  #[surfaces,thetas,zetas,vpars] = np.meshgrid(s_lin,theta_lin,zeta_lin,vpar_lin)

  # gauss quadrature
  s_lin,s_weights = gauss_quadrature_nodes_coeffs(ns,s_min,s_max)
  theta_lin,theta_weights = gauss_quadrature_nodes_coeffs(ntheta,0,2*np.pi)
  zeta_lin,zeta_weights = gauss_quadrature_nodes_coeffs(nzeta,0,2*np.pi/tracer.surf.nfp)
  vpar_lin,vpar_weights = gauss_quadrature_nodes_coeffs(nvpar,-V_MAX,V_MAX)
  [surfaces,thetas,zetas,vpars] = np.meshgrid(s_lin,theta_lin,zeta_lin,vpar_lin,indexing='ij')
  [w1,w2,w3,w4] = np.meshgrid(s_weights,theta_weights,zeta_weights,vpar_weights,indexing='ij')
  quad_weights = w1*w2*w3*w4

  # build a mesh
  stz_inits = np.zeros((ns*ntheta*nzeta*nvpar, 3))
  stz_inits[:, 0] = surfaces.flatten()
  stz_inits[:, 1] = thetas.flatten()
  stz_inits[:, 2] = zetas.flatten()
  vpar_inits = vpars.flatten()
  # radial likelihood
  likelihood = radial_sampler._pdf(stz_inits[:,0])
  likelihood *= (1/(2*np.pi))*(tracer.surf.nfp/(2*np.pi))*(1/(2*V_MAX))

elif sampling_type == "grid":
  # grid over (theta,phi,vpar) for a fixed surface label
  s_label = float(sampling_level)

  ## simpson
  #theta_lin = np.linspace(0, 2*np.pi, ntheta)
  #zeta_lin = np.linspace(0,2*np.pi/tracer.surf.nfp, nzeta)
  #vpar_lin = np.linspace(-V_MAX,V_MAX,nvpar)
  #[thetas,zetas,vpars] = np.meshgrid(theta_lin,zeta_lin,vpar_lin)

  # gauss quadrature
  theta_lin,theta_weights = gauss_quadrature_nodes_coeffs(ntheta,0,2*np.pi)
  zeta_lin,zeta_weights = gauss_quadrature_nodes_coeffs(nzeta,0,2*np.pi/tracer.surf.nfp)
  vpar_lin,vpar_weights = gauss_quadrature_nodes_coeffs(nvpar,-V_MAX,V_MAX)
  [thetas,zetas,vpars] = np.meshgrid(theta_lin,zeta_lin,vpar_lin,indexing='ij')
  [w1,w2,w3] = np.meshgrid(theta_weights,zeta_weights,vpar_weights,indexing='ij')
  quad_weights = w1*w2*w3

  # build a mesh
  stz_inits = np.zeros((ntheta*nzeta*nvpar, 3))
  stz_inits[:, 0] = s_label
  stz_inits[:, 1] = thetas.flatten()
  stz_inits[:, 2] = zetas.flatten()
  vpar_inits = vpars.flatten()
  # constant likelihood
  likelihood = np.ones(len(vpar_inits))
  likelihood *= (1/(2*np.pi))*(tracer.surf.nfp/(2*np.pi))*(1/(2*V_MAX))

elif sampling_type == "SAA":
  stz_inits,vpar_inits = get_random_points(sampling_level)


"""
Define the optimization objective
"""

def objective(x):
  """
  Two objectives for minimization:
  
  1. expected energy retatined
    f = E[3.5*np.exp(-2*c_times/tmax)]
  2. expected confinement time
    f = tmax - E[c_times]

  x: array,vmec configuration variables
  tmax: float, max trace time
  """
  if sampling_type == "random":
    stzs,vpars = get_random_points(sampling_level)
  else:
    stzs = np.copy(stz_inits)
    vpars = np.copy(vpar_inits)

  c_times = tracer.compute_confinement_times(x,stzs,vpars,tmax)

  if np.any(~np.isfinite(c_times)):
    # vmec failed here; return worst possible value
    c_times = np.zeros(len(vpars))


  if objective_type == "mean_energy": 
    # minimize energy retained by particle
    feat = 3.5*np.exp(-2*c_times/tmax)
  elif objective_type == "mean_time": 
    # expected confinement time
    feat = tmax-c_times


  # now perform the averaging/quadrature
  if sampling_type in ["SAA", "random"]:
    # sample average
    res = np.mean(feat)
    loss_frac = np.mean(c_times<tmax)
  elif sampling_type == "grid" and sampling_level == "full":
    # gauss quadrature
    int0 = feat*likelihood
    int0 = int0.reshape((ns,ntheta,nzeta,nvpar))
    int0 = int0*quad_weights
    res = np.sum(int0)

    # loss frac for printing
    loss_frac = c_times<tmax
    int0 = loss_frac*likelihood
    int0 = int0.reshape((ns,ntheta,nzeta,nvpar))
    int0 = int0*quad_weights
    loss_frac = np.sum(int0)

    # simpson
    #int0 = feat*likelihood
    #int0 = int0.reshape((ns,ntheta,nzeta,nvpar))
    #int1 = simpson(int0,vpar_lin,axis=-1)
    #int2 = simpson(int1,zeta_lin,axis=-1)
    #int3 = simpson(int2,theta_lin,axis=-1)
    #res = simpson(int3,s_lin,axis=-1)
  elif sampling_type == "grid":
    # gauss quadrature
    int0 = feat*likelihood
    int0 = int0.reshape((ntheta,nzeta,nvpar))
    int0 = int0*quad_weights
    res = np.sum(int0)

    # loss fraction for printing
    loss_frac = c_times<tmax
    int0 = loss_frac*likelihood
    int0 = int0.reshape((ntheta,nzeta,nvpar))
    int0 = int0*quad_weights
    loss_frac = np.sum(int0)

    # simpson
    #int0 = feat*likelihood
    #int0 = int0.reshape((ntheta,nzeta,nvpar))
    #int1 = simpson(int0,vpar_lin,axis=-1)
    #int2 = simpson(int1,zeta_lin,axis=-1)
    #res = simpson(int2,theta_lin,axis=-1)

  if rank == 0:
    print('obj:',res,'P(loss):',loss_frac)
  sys.stdout.flush()

  return res

evw = EvalWrapper(objective,dim_x,1)



if rank == 0:
  print(f"optimizing with tmax = {tmax}")


# optimize
if method == "cobyla":
  rhobeg = max_step
  rhoend = min_step
  aspect_constraint = pdfo_nlc(aspect_ratio,-np.inf,aspect_target)
  iota_constraint = pdfo_nlc(rotational_transform,iota_target,iota_target)
  mirror_constraint = pdfo_nlc(B_field,B_lb,B_ub)
  #mirror_constraint = pdfo_nlc(B_field_volavg_con,B_lb,B_ub)
  constraints = [aspect_constraint,mirror_constraint]
  if constrain_iota:
    constraints.append(iota_constraint)

  res = pdfo(evw, x0, method='cobyla',constraints=constraints,options={'maxfev': maxfev, 'ftarget': ftarget,'rhobeg':rhobeg,'rhoend':rhoend})
  xopt = np.copy(res.x)

elif method == "bobyqa":
  rhobeg = max_step
  rhoend = min_step

  def penalty_obj(x):
    """penalty formulation for bobyqa"""
    B = B_field(x)
    #B = B_field_volavg_con(x)
    obj = evw(x)
    # B_lb <= modB <= B_ub
    c_mirr_ub = np.sum(np.maximum(B - B_ub, 0.0)**2)
    c_mirr_lb = np.sum(np.maximum(B_lb - B, 0.0)**2)
    # aspect <= aspect_target
    asp = aspect_ratio(x)
    c_asp = max([asp-aspect_target,0.0])**2
    # iota constraint iota = iota_target
    #iota = rotational_transform(x)
    if constrain_iota:
      iota = np.mean(tracer.vmec.wout.iotas[1:]) # faster computation of iota
      c_iota = (iota-iota_target)**2
    else:
      c_iota = 0.0
    ret = obj + c_asp + (c_mirr_ub + c_mirr_lb) + c_iota
    if rank == 0:
      print('p-obj:',ret,'asp',asp,'iota',iota,'c_mirr_ub',c_mirr_ub,'c_mirr_lb',c_mirr_lb)
      #print("")
    return ret
  res = pdfo(penalty_obj, x0, method='bobyqa',options={'maxfev': maxfev, 'ftarget': ftarget,'rhobeg':rhobeg,'rhoend':rhoend})
  xopt = np.copy(res.x)


#elif method == 'snobfit':
#  # snobfit
#  bounds = np.vstack((x_lb,x_ub)).T
#  res, _ = skq_minimize(evw, x0, bounds, maxfev, method='SnobFit')
#  xopt = np.copy(res.optpar)
#
#elif method == "diff_evol":
#  # differential evolution
#  bounds = np.vstack((x_lb,x_ub)).T
#  popsize = 10 # population is popsize*dim_x individuals
#  maxiter = int(maxfev/dim_x/popsize)
#  res = differential_evolution(evw,bounds=bounds,popsize=popsize,maxiter=maxiter,x0=x0)
#  xopt = np.copy(res.x)

elif method == "nelder":
  init_simplex = np.zeros((dim_x+1,dim_x))
  init_simplex[0] = np.copy(x0)
  init_simplex[1:] = np.copy(x0 + max_step*np.eye(dim_x))
  def penalty_obj(x):
    obj = evw(x)
    asp = aspect_ratio(x)
    return obj + 1000*np.max([asp-aspect_target,0.0])**2
  # nelder-mead
  xatol = min_step # minimum step size
  res = sp_minimize(penalty_obj,x0,method='Nelder-Mead',
              options={'maxfev':maxfev,'xatol':xatol,'initial_simplex':init_simplex})
  xopt = np.copy(res.x)

#elif method == "sidpsm":
#  def penalty_obj(x):
#    obj = evw(x)
#    if objective_type == "mean_energy" and obj >= 3.5:
#      return np.inf
#    elif objective_type == "mean_time" and obj >=tmax:
#      return np.inf
#    asp = aspect_ratio(x)
#    return obj + 1000*np.max([asp-aspect_target,0.0])**2
#  sid = SIDPSM(penalty_obj,x0,max_eval=maxfev,delta=max_step,delta_min=min_step,delta_max=max_step)
#  res = sid.solve()
#  xopt = np.copy(res['x'])


# evaluate the configuration
if sampling_type == "random":
  stz_inits,vpar_inits = get_random_points(sampling_level)
c_times_opt = tracer.compute_confinement_times(xopt,stz_inits,vpar_inits,tmax)

tracer.surf.x = np.copy(xopt)
aspect_opt = tracer.vmec.aspect()

# out of sample performance
stz_rand,vpar_rand = get_random_points(sampling_level)
c_times_out_of_sample = tracer.compute_confinement_times(xopt,stz_rand,vpar_rand,tmax)

if rank == 0:
  print('aspect(xopt)',aspect_opt)
  print('E[c_time(xopt)]',np.mean(c_times_out_of_sample))
  print('Loss fraction',np.mean(c_times_out_of_sample<tmax))
  print('E[Energy]',np.mean(3.5*np.exp(-2*c_times_out_of_sample/tmax)))

# save results
if rank == 0:
  print(res)
  outfile = f"./data_opt_{vmec_label}_{objective_type}_{sampling_type}_surface_{sampling_level}_tmax_{tmax}_{method}_mmode_{max_mode}_iota_{iota_target}.pickle"
  outdata = {}
  outdata['X'] = evw.X
  outdata['FX'] = evw.FX
  outdata['xopt'] = xopt
  outdata['aspect_opt'] = aspect_opt
  outdata['c_times_opt'] = c_times_opt
  outdata['c_times_out_of_sample'] = c_times_out_of_sample
  outdata['aspect_target'] = aspect_target
  outdata['iota_target'] = iota_target
  outdata['constrain_iota'] = constrain_iota
  outdata['major_radius'] = major_radius
  #outdata['minor_radius'] =  minor_radius
  outdata['target_volavgB'] = target_volavgB
  outdata['vmec_input'] = vmec_input
  outdata['max_mode'] = max_mode
  outdata['vmec_input'] = vmec_input
  outdata['warm_start_file'] = warm_start_file 
  outdata['objective_type'] = objective_type
  outdata['sampling_type'] = sampling_type
  outdata['sampling_level'] = sampling_level
  outdata['method'] = method
  outdata['maxfev'] = maxfev
  outdata['max_step'] = max_step
  outdata['min_step'] = min_step
  outdata['tracing_tol'] = tracing_tol
  outdata['interpolant_degree'] = interpolant_degree
  outdata['interpolant_level'] = interpolant_level
  outdata['bri_mpol'] = bri_mpol
  outdata['bri_ntor'] = bri_ntor
  #outdata['stz_inits'] = stz_inits
  #outdata['vpar_inits'] = vpar_inits
  outdata['tmax'] = tmax
  pickle.dump(outdata,open(outfile,"wb"))
