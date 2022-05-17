import numpy as np
from scipy.interpolate import RegularGridInterpolator,interp1d
#from eqtools.trispline import Spline as eqSpline
from pyevtk.hl import gridToVTK 
from scipy import integrate


class STELLA:
  """
  A semi-lagrangian method for solving the guiding center advection equation
  in standard cylindrical coordinates (r,phi,z) and vparallel.
    u_t + G @ grad(u) = 0
  where u is a function u(r,phi,z,vpar,t) and G is a function G(r,phi,z,vpar)
  that represents the rhs of the vacuum guiding center equations 
  in three spatial dimensions. 
  
  We assume stellarator symmetry, so that G is periodic across field periods.

  Potential problems
  - Periodic boundary conditions are unphysical for vpar.
  - Our method does not conserve mass. For this we should use a mass-corrector method.
  - The boundary of the plasma may be problematic because it represents a discontinuity in
    the density function. So we should have high resolution grid/interpolation there. 
    Furthermore, there may be problems with the fractal like structure generated by the ODE.
  """
  
  def __init__(self,u0,B,gradAbsB,
    rmin,rmax,n_r,n_phi,nfp,
    zmin,zmax,n_z,
    vparmin,vparmax,n_vpar,
    dt,tmax,integration_method,
    mesh_type="uniform",
    include_drifts=True,
    interp_type="linear",
    outputdir="./plot_data"):

    # initial distribution u0(r,phi,z,vpar)
    self.u0 = u0
    # B field function B(x,y,z)
    self.B = B
    # gradient of ||B|| as a function of (x,y,z)
    self.gradAbsB = gradAbsB

    # number field periods
    self.nfp = nfp 
    # toroidal mesh sizes
    self.n_r = n_r
    self.n_phi = n_phi
    self.n_z = n_z
    self.n_vpar = n_vpar
    #self.dr = dr
    #self.dphi = dphi
    #self.dz = dz
    #self.dvpar = dvpar
    self.dt = dt
    # mesh bounds
    self.rmin = rmin
    # TODO: make a system to automatically set the domain bounds
    self.rmax = rmax 
    self.phimin = 0.0
    self.phimax = 2*np.pi/nfp 
    self.zmin = zmin
    self.zmax = zmax
    self.vparmin = vparmin
    self.vparmax = vparmax
    self.tmin = 0.0
    self.tmax = tmax

    # for backwards integration
    assert integration_method in ['euler','midpoint','rk4']
    self.integration_method = integration_method

    # drift terms
    self.include_drifts = include_drifts

    # for meshing
    assert mesh_type in ['uniform','chebyshev']
    self.mesh_type = mesh_type

    # for interpolation
    assert interp_type in ['linear', 'cubic']
    self.interp_type = interp_type

    # for plotting
    self.outputdir = outputdir

    self.PROTON_MASS = 1.67262192369e-27  # kg
    self.NEUTRON_MASS = 1.67492749804e-27  # kg
    self.ELEMENTARY_CHARGE = 1.602176634e-19  # C
    self.ONE_EV = 1.602176634e-19  # J
    self.ALPHA_PARTICLE_MASS = 2*self.PROTON_MASS + 2*self.NEUTRON_MASS
    self.ALPHA_PARTICLE_CHARGE = 2*self.ELEMENTARY_CHARGE
    self.FUSION_ALPHA_PARTICLE_ENERGY = 3.52e6 * self.ONE_EV # Ekin
    self.FUSION_ALPHA_SPEED_SQUARED = 2*self.FUSION_ALPHA_PARTICLE_ENERGY/self.ALPHA_PARTICLE_MASS

  def cheby_lin(self,a,b,n):
    """
    Create a one dimensional grid using chebyshev nodes,
    see https://en.wikipedia.org/wiki/Chebyshev_nodes
    for a definition.

    input: 
    a,b: float, interval bounds
    n: int, number of grid points

    return:
    x: 1d array, entries are chebyshev nodes.
    """
    k = np.arange(1,n+1,1)
    arg = np.pi*(2*k-1)/2/n
    ret = 0.5*(a+b) + 0.5*(b-a)*np.cos(arg)
    return np.copy(ret[::-1])

  def cyl_to_cart(self,r_phi_z):
    """ cylindrical to cartesian coordinates 
    input:
    r_phi_z: (N,3) array of points (r,phi,z)

    return
    xyz: (N,3) array of point (x,y,z)
    """
    r = r_phi_z[:,0]
    phi = r_phi_z[:,1]
    z = r_phi_z[:,2]
    return np.vstack((r*np.cos(phi),r*np.sin(phi),z)).T

  #def jac_cyl_to_cart(self,r,phi,z):
  # """
  # jacobian of (x,y,z) with respect to (r,phi,z) evaluated
  # at r,phi,z.

  # J = [[cos(phi), -r*sin(phi),0]
  #      [sin(phi),r*cos(phi),0]
  #      [0,0,1]]

  # return 
  # J: (3,3) jacobian matrix
  # """
  # J = np.array([[np.cos(phi),-r*np.sin(phi),0.],[np.sin(phi),r*np.cos(phi),0.0],[0,0,1.0]])
  # return J

  def jac_cart_to_cyl(self,r_phi_z,D):
   """
   Compute the jacobian vector product of the jacobian of the coordinate 
   transformation from cartesian to cylindrical with a set of vectors.
   The Jacobian 
     J = [[cos(phi), sin(phi),0]
          [-sin(phi),cos(phi),0]/r
          [0,0,1]]
   is evaluated at the points in r_phi_z, then the product is computed against
   the vectors in D.

   input:
   r_phi_z: (N,3) array of points in cylindrical coordinates, (r,phi,z)
   D: (N,3) array of vectors in xyz coordinates to compute the directional derivatives.
   return 
   JD: (N,3) array of jacobian vector products, J @ D
   """
   #J = np.array([[np.cos(phi),np.sin(phi),0.],[-np.sin(phi)/r,np.cos(phi)/r,0.0],[0,0,1.0]])
   r = r_phi_z[:,0]
   phi = r_phi_z[:,1]
   JD = np.vstack((np.cos(phi)*D[:,0] + np.sin(phi)*D[:,1],
                  (-np.sin(phi)*D[:,0] + np.cos(phi)*D[:,1])/r,
                  D[:,2])).T
   return JD

  def GC_rhs(self,X):
    """ 
    Right hand side of the vacuum guiding center equations in 
    cylindrical coordinates.

    input:
    X: (N,4) array of points in cylindrical coords (r,phi,z,vpar)
    
    return
    (N,4) array, time derivatives [dr/dt,dphi/dt,dz/dt,dvpar/dt]
    """
    # extract values
    r_phi_z = np.copy(X[:,:-1])
    vpar = np.copy(X[:,-1])

    # convert to xyz
    xyz = self.cyl_to_cart(r_phi_z)

    # dont compute repeated points on grid
    xyz_for_B = xyz[::self.n_vpar]

    # compute B on grid
    Bb = self.B(xyz_for_B)
    Bb = np.copy(np.repeat(Bb,self.n_vpar,axis=0))
    #Bb = self.B(xyz) # shape (N,3)

    # compute Bg on grid
    Bg = self.gradAbsB(xyz_for_B)
    Bg = np.copy(np.repeat(Bg,self.n_vpar,axis=0))
    #Bg = self.gradAbsB(xyz) # shape (N,3)

    # field values
    B = np.linalg.norm(Bb,axis=1) # shape (N,)
    b = (Bb.T/B).T # shape (N,3)

    vperp_squared = self.FUSION_ALPHA_SPEED_SQUARED - vpar**2 # shape (N,)
    c = self.ALPHA_PARTICLE_MASS /self.ALPHA_PARTICLE_CHARGE/B/B/B # shape (N,)
    mu = vperp_squared/2/B

    # compute d/dt (x,y,z); shape (N,3)
    if self.include_drifts:
      dot_xyz = ((b.T)*vpar + c*(vperp_squared/2 + vpar**2) * np.cross(Bb,Bg).T).T
    else:
      dot_xyz = ((b.T)*vpar).T # no drift terms

    # compute d/dt (r,phi,z); shape (N,3)
    dot_rphiz =self.jac_cart_to_cyl(r_phi_z,dot_xyz)

    # compute d/dt (vpar); shape (N,)
    dot_vpar = -mu*np.sum(b * Bg,axis=1)

    # compile into a vector; shape (N,4)
    dot_state = np.hstack((dot_rphiz,np.reshape(dot_vpar,(-1,1))))
    return dot_state
    
  def startup(self):
    """
    Evaluate u0 over the mesh.
    """

    if self.mesh_type == "uniform":
      # mesh spacing
      r_lin = np.linspace(self.rmin,self.rmax,self.n_r)
      phi_lin = np.linspace(self.phimin,self.phimax,self.n_phi)
      z_lin = np.linspace(self.zmin,self.zmax,self.n_z)
      vpar_lin = np.linspace(self.vparmin,self.vparmax,self.n_vpar)
    elif self.mesh_type == "chebyshev":
      # mesh spacing
      r_lin = self.cheby_lin(self.rmin,self.rmax,self.n_r)
      phi_lin = np.linspace(self.phimin,self.phimax,self.n_phi) # phi is periodic
      z_lin = self.cheby_lin(self.zmin,self.zmax,self.n_z)
      vpar_lin = self.cheby_lin(self.vparmin,self.vparmax,self.n_vpar)
      # the mesh will not hit the bounds, so redefine the bounds
      self.rmin = np.min(r_lin)
      self.rmax = np.max(r_lin)
      self.phimin = np.min(phi_lin)
      self.phimax = np.max(phi_lin)
      self.zmin = np.min(z_lin)
      self.zmax = np.max(z_lin)
      self.vparmin = np.min(vpar_lin)
      self.vparmax = np.max(vpar_lin)
   

    # build the grid
    r_grid,phi_grid,z_grid,vpar_grid = np.meshgrid(r_lin,phi_lin,
                             z_lin,vpar_lin,indexing='ij')

    # reshape the grid to points
    X = np.vstack((np.ravel(r_grid),np.ravel(phi_grid),
            np.ravel(z_grid),np.ravel(vpar_grid))).T

    # compute u0
    U_grid = self.u0(X)
    U_grid = np.reshape(U_grid,np.shape(r_grid))

    # store the value of the density in grid shaping
    self.U_grid = np.copy(U_grid)

    # save a 2d-array of grid points
    self.X = np.copy(X)

    # save the spacing
    self.r_lin = r_lin
    self.phi_lin = phi_lin
    self.z_lin = z_lin
    self.vpar_lin = vpar_lin
    return 

  def write_mesh_vtk(self,tau=0.0):
    """
    Write the probability density to a vtk file. 
    
    Write U_grid to vtk files by taking slices across
    the vpar component. The vtk files should show the 
    value of the density at over the spatial domain with 
    a fixed value of vpar. By default, files are dropped
    for each value of vpar.
    
    tau: time variable for naming files.
    """
    # build the grid in spatial components
    r_grid,phi_grid,z_grid = np.meshgrid(self.r_lin,self.phi_lin,
                             self.z_lin,indexing='ij')
    r_phi_z = np.vstack((np.ravel(r_grid),np.ravel(phi_grid),
            np.ravel(z_grid))).T
    xyz = self.cyl_to_cart(r_phi_z)
    X = np.reshape(xyz[:,0],np.shape(r_grid))
    Y = np.reshape(xyz[:,1],np.shape(r_grid))
    Z = np.reshape(xyz[:,2],np.shape(r_grid))

    # get the current density
    U_grid = np.copy(self.U_grid)

    for ii,vpar in enumerate(self.vpar_lin):
      # take a slice of U_grid in vpar direction
      U_slice = U_grid[:,:,:,ii]
      # reshape like spatial grid
      U_slice = np.copy(np.reshape(U_slice,np.shape(r_grid)))
      # write the vtk
      path = self.outputdir + f"/u_mesh_time_{tau}_vpar_{ii}"
      gridToVTK(path, X,Y,Z,pointData= {'u':U_slice})

  def write_spatial_marginal_vtk(self,tau=0.0):
    """
    Write the marginal probability density u(r,phi,z) to a vtk file. 

    tau: time variable for naming files.
    """
    # build the grid in spatial components
    r_grid,phi_grid,z_grid = np.meshgrid(self.r_lin,self.phi_lin,
                             self.z_lin,indexing='ij')
    r_phi_z = np.vstack((np.ravel(r_grid),np.ravel(phi_grid),
            np.ravel(z_grid))).T
    xyz = self.cyl_to_cart(r_phi_z)
    X = np.reshape(xyz[:,0],np.shape(r_grid))
    Y = np.reshape(xyz[:,1],np.shape(r_grid))
    Z = np.reshape(xyz[:,2],np.shape(r_grid))
    # get the spatial marginal
    U_marg = self.compute_spatial_marginal()
    # transform density to cartesian; det(jac) = 1/r
    U_marg = U_marg*(1.0/r_grid)
    # write the vtk
    path = self.outputdir + f"/u_spatial_marginal_time_{tau:08d}"
    gridToVTK(path, X,Y,Z,pointData= {'u':U_marg})

    
  def backstep(self):
    """
    Backwards integrate along the characteristic from time
    t_np1 to time t_n to find the deparature points. 
    Return the departure points (r,theta,phi,vpar).

    We have implemented three choices of methods: Euler's method, the
    Explicit Midpoint method, and the 4th order Runge Kutta. 
 
    return 
    X: (...,4) array of departuare points
    """
    
    # get the 2d array of grid points
    X = np.copy(self.X)

    # backwards integrate the points
    if self.integration_method == "euler":
      G =  self.GC_rhs(X)
      Xtm1 = np.copy(X)
      Vtm1 = np.copy(X)
      # step in spatial
      Xtm1[:,:-1] = X[:,:-1]- self.dt*G[:,:-1]
      # step in vpar
      Vtm1[:,-1] = X[:,-1]- self.dt*G[:,-1]/2
    elif self.integration_method == "midpoint":
      G =  self.GC_rhs(X)
      Xstar = np.copy(X)
      Vstar = np.copy(X)
      # compute midpoints
      Xstar[:,:-1] = np.copy(X[:,:-1] - self.dt*G[:,:-1]/2)
      Vstar[:,-1]  = np.copy(X[:,-1] - self.dt*G[:,-1]/4)
      # take step in spatial
      # TODO: improve efficiency by only computing the GC equations for spatial.
      G1 =  self.GC_rhs(Xstar)
      Xtm1 = np.copy(X)
      Xtm1[:,:-1] = np.copy(X[:,:-1] - self.dt*G1[:,:-1])
      # take step in vpar
      # TODO: improve efficiency by reusing B from first G computation
      # TODO: improve efficiency by only computing the GC equations for vpar.
      G2 =  self.GC_rhs(Vstar) 
      Vtm1 = np.copy(X)
      Vtm1[:,-1] = np.copy(X[:,-1] - self.dt*G2[:,-1]/2)

    elif self.integration_method == "rk4":
      print("rk4 not functional")
      quit()
      # TODO: not working. not vectorized either.
      G = np.array([self.GC_rhs(*xx) for xx in X])
      k1 = np.copy(self.dt*G)
      G = np.array([self.GC_rhs(*xx) for xx in X+k1/2])
      k2 = np.copy(self.dt*G)
      G = np.array([self.GC_rhs(*xx) for xx in X+k2/2])
      k3 = np.copy(self.dt*G)
      G = np.array([self.GC_rhs(*xx) for xx in X+k3])
      k4 = np.copy(self.dt*G)
      Xtm1 = np.copy(X -(k1+2*k2+2*k3+k4)/6)

    return Xtm1,Vtm1

  def interpolate_linear(self,X):
    """
    Interpolate the value of u(r,theta,phi,vpar) from grid points to 
    the set of points X.

    input:
    X: (N,4) array, points at which to interpolate the value of u(x,y,z,vpar)

    return: 
    U: (N,) array, interpolated values of u(state) for state in X.
    """
    interpolator = RegularGridInterpolator((self.r_lin,self.phi_lin,
                   self.z_lin,self.vpar_lin), self.U_grid)
    UX = interpolator(X)
    return np.copy(UX)

  #def interpolate_spatial(self,X_feas,idx_feas):
  #  """
  #  Interpolate over the vpar dimension. We assume that the first three
  #  columns of V, (r,phi,z), are grid values. This allows us to only
  #  interpolate over the fourth dimension, vpar.

  #  input:
  #  X: (N,4) array, points at which to interpolate the value of u(x,y,z,vpar)

  #  return: 
  #  U: (N,) array, interpolated values of u(state) for state in X.
  #  """
  #  UX = np.zeros(len(X))
  #  block_size = self.n_r*self.n_phi*self.n_vpar
  #  for ii in range(self.n_vpar):
  #    # integrate over all grid points with the same (r,phi,z) values
  #    interpolator = eqSpline((self.r_lin,self.phi_lin,
  #                 self.z_lin), self.U_grid[:,:,:,ii])
  #    idx = ii*block_size
  #    UX[idx:idx + block_size] = interpolator(X[idx:idx + block_size,:-1])
  #  return np.copy(UX)

  def interpolate_vpar(self,V):
    """
    Interpolate over the vpar dimension. We assume that the first three
    columns of V, (r,phi,z), are grid values. This allows us to only
    interpolate over the fourth dimension, vpar.

    input:
    X: (N,4) array, points at which to interpolate the value of u(x,y,z,vpar)

    return: 
    U: (N,) array, interpolated values of u(state) for state in X.
    """
    UX = np.zeros(len(V))
    for ii in range(self.n_r):
      for jj in range(self.n_phi):
        for kk in range(self.n_z):
          # integrate over all grid points with the same (r,phi,z) values
          # TODO: allow for linear interpolation of vpar here as well.
          interpolator = interp1d(self.vpar_lin,self.U_grid[ii,jj,kk,:],kind='cubic')
          idx = (ii*self.n_phi*self.n_z + jj *self.n_z + kk)*self.n_vpar
          UX[idx:idx + self.n_vpar] = interpolator(V[idx:idx + self.n_vpar,-1])
    return np.copy(UX)

  def update_U_grid(self,UX):
    """
    Take a forward timestep in the advection equation
    in u to compute the value of u at the grid points at
    time t+1 from the interpolated points. 

    For our problem, U does not change along characteristics.
    So the value of U at the grid points at time t+1 is the
    value of U at the interpolated points. 
    """
    # reshape the list of evals into a grid shape
    self.U_grid = np.copy(np.reshape(UX,np.shape(self.U_grid)))
    return 

  def apply_boundary_conds(self,X,V):
    """
    Apply the boundary conditions. 
    If a departure point is outside of the mesh, then we have to apply
    boundary conditions to ensure correctness of the solve. We return the set
    of points that are valid for interpolation.

    If a departure point is outside of the mesh in the r or z directions, then
    it the density of the departure point is zero. This is because particles cannot
    enter the device from outside of the device walls.

    phi has periodic boundary conditions on its domain.

    We give vpar periodic boundary conditions. This is unphysical, as it passes
    probability mass between particles with velocity >=vparmax to particles with
    velocity <=vparmin, and vice versa. To avoid the effects of this boundary 
    condition, set vparmin and vparmax to be Large! We can theoretically bound the
    max velocity of particles within a given time, so the vpar bounds can be set
    based on that. Alternatively we can track mu instead of vpar.

    input:
      X: (N,dim_state) array of departure points
    return:
      X_feas: (M,dim_state) array of departure points which are contained within
         the meshed volume. Any points in X with r or z outside of their bounds
         are dropped. The phi,vpar directions of a given point which are violating
         their bounds are corrected to satisfy the periodic boundary conditions. Thus
         X_feas is the set of departure points that are within the mesh.
      idx_feas: boolean array indicating the indexes of X such that X_feas = X[idx_feas].
    """
    # get indexes of points with r,z in the meshed volume
    r_in = np.logical_and(X[:,0]>=self.rmin,X[:,0]<=self.rmax)
    z_in = np.logical_and(X[:,2]>=self.zmin,X[:,2]<=self.zmax)
    idx_feas = np.logical_and(r_in,z_in)
    X_feas = np.copy(X[idx_feas])
    # phi periodic on [phimin,phimax)... (assumes phimin = 0)
    X_feas[:,1] %= self.phimax # phi in [-phimax,phimax]
    X_feas[:,1][X_feas[:,1]<0] += self.phimax # correct negatives

    # vpar periodic
    vpardiff = self.vparmax - self.vparmin
    idx_up =  V[:,3] > self.vparmax
    V[:,3][idx_up] = self.vparmin + (V[:,3][idx_up]-self.vparmax)%vpardiff
    idx_down =  V[:,3] < self.vparmin
    V[:,3][idx_down] = self.vparmax - (self.vparmin-V[:,3][idx_down])%vpardiff

    return np.copy(X_feas),idx_feas,np.copy(V)
    

  def solve(self,classifier=None):
    """
    Perform the pde solve.
    """
    import time
    # build the mesh and evalute U
    t0 = time.time()
    self.startup()
    print('startup time',time.time()-t0)

    # only backwards integrate once b/c time independent
    t0 = time.time()
    X,V = self.backstep()
    print('backstep time',time.time()-t0)

    # allocate space for density values
    UX = np.zeros(len(X))

    # collect departure points within the meshed volume
    t0 = time.time()
    X_feas,idx_feas,V = self.apply_boundary_conds(X,V)
    idx_infeas = (idx_feas == False)
    print('bndry cond time',time.time()-t0)

    times = np.arange(self.tmin,self.tmax,self.dt)
    for n_step,tt in enumerate(times):
      if n_step%1 == 0:
        self.write_spatial_marginal_vtk(tau=n_step)
        if classifier is not None:
          print('t = ',tt,'p = ',self.compute_plasma_probability_mass(classifier))
        else:
          print('t = ',tt)

      # intepolate the values of the departure points
      # this step assumes the dirichlet boundary conditions
      # i.e. that UX[not idx_feas] == 0
      #UX = self.interpolate(V)

      # take half step in vpar
      if self.interp_type == "linear":
        UX = self.interpolate_linear(V)
      elif self.interp_type == "cubic":
        UX = self.interpolate_vpar(V)
      self.update_U_grid(UX)

      # take full step in spatial
      UX[idx_feas] = self.interpolate_linear(X_feas)
      UX[idx_infeas] = 0.0 # departure points outside of the mesh
      self.update_U_grid(UX)

      # take second half step in vpar
      if self.interp_type == "linear":
        UX = self.interpolate_linear(V)
      elif self.interp_type == "cubic":
        UX = self.interpolate_vpar(V)
      self.update_U_grid(UX)

    return self.U_grid

  def compute_spatial_marginal(self):
    """ 
    Compute the marginal density over the spatial variables by 
    integrating over the vpar domain.
    return the marginal u(r,phi,z) as a 3d meshgrid.
    """
    #U_marg = integrate.simpson(self.U_grid,x=self.vpar_lin,axis=-1)
    U_marg = integrate.trapezoid(self.U_grid,x=self.vpar_lin,axis=-1)
    return U_marg

  def compute_plasma_probability_mass(self,classifier):
    """
    Compute the loss fraction by integrating the probability density
    over the plasma boundary in the meshed region. Since
    we model the probability density over one field period, you 
    should multiply by the probability mass by the number of field 
    periods nfp to get the probility over the entire plasma.

    input: simsopt surface classifier as a function of x,y,z
    """
    # integrate over vpar
    U_marg = np.copy(self.compute_spatial_marginal())

    # build the grid in spatial components
    r_grid,phi_grid,z_grid = np.meshgrid(self.r_lin,self.phi_lin,
                             self.z_lin,indexing='ij')
    r_phi_z = np.vstack((np.ravel(r_grid),np.ravel(phi_grid),
            np.ravel(z_grid))).T
    xyz = self.cyl_to_cart(r_phi_z)

    # zero out U_marg where r,phi,z is not in plasma
    c= np.array([classifier.evaluate(x.reshape((1,-1)))[0].item() for x in xyz])
    idx_infeas = c < 0
    idx_infeas = np.copy(np.reshape(idx_infeas,np.shape(U_marg)))
    U_marg[idx_infeas] = 0.0

    # now integrate U_marg
    #U_marg = integrate.simpson(U_marg,x=self.z_lin,axis=-1)
    #U_marg = integrate.simpson(U_marg,x=self.phi_lin,axis=-1)
    #prob = integrate.simpson(U_marg,x=self.r_lin,axis=-1)
    U_marg = integrate.trapezoid(U_marg,x=self.z_lin,axis=-1)
    U_marg = integrate.trapezoid(U_marg,x=self.phi_lin,axis=-1)
    prob = integrate.trapezoid(U_marg,x=self.r_lin,axis=-1)
  
    return prob
