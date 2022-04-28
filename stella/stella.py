import numpy as np
from scipy.interpolate import RegularGridInterpolator


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
    rmin,rmax,dr,dphi,nfp,
    zmin,zmax,dz,
    vparmin,vparmax,dvpar,
    dt,tmax,integration_method):
    """
    """
    # initial distribution function u0(r,phi,z,vpar)
    self.u0 = u0
    # B field function B(r,phi,z)
    self.B = B
    # gradient of ||B|| as a function of (r,phi,z)
    self.gradAbsB = gradAbsB

    # number field periods
    self.nfp = nfp 
    # toroidal mesh sizes
    self.dr = dr
    self.dphi = dphi
    self.dz = dz
    self.dvpar = dvpar
    self.dt = dt
    # mesh bounds
    self.rmin = rmin
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
 
    self.PROTON_MASS = 1.67262192369e-27  # kg
    self.NEUTRON_MASS = 1.67492749804e-27  # kg
    self.ELEMENTARY_CHARGE = 1.602176634e-19  # C
    self.ONE_EV = 1.602176634e-19  # J
    self.ALPHA_PARTICLE_MASS = 2*self.PROTON_MASS + 2*self.NEUTRON_MASS
    self.ALPHA_PARTICLE_CHARGE = 2*self.ELEMENTARY_CHARGE
    self.FUSION_ALPHA_PARTICLE_ENERGY = 3.52e6 * self.ONE_EV # Ekin
    self.FUSION_ALPHA_SPEED_SQUARED = 2*self.FUSION_ALPHA_PARTICLE_ENERGY/self.ALPHA_PARTICLE_MASS

  def jac_cart_to_cyl(self,r,phi,z):
   """
   jacobian of (r,phi,z) with respect to (x,y,z) evaluated
   at r,phi,z.

   J = [[cos(phi), sin(phi),0]
        [-sin(phi),cos(phi),0]/r
        [0,0,1]]

   return 
   J: (3,3) jacobian matrix
   """
   J = np.array([[np.cos(phi),np.sin(phi),0.],[-np.sin(phi)/r,np.cos(phi)/r,0.0],[0,0,1.0]])
   return J

  def GC_rhs(self,r,phi,z,vpar):
    """ 
    Right hand side of the vacuum guiding center equations in 
    cylindrical coordinates.

    input:
    r,phi,z,vpar: floats
    
    return
    (4,) array, time derivatives [dr/dt,dphi/dt,dz/dt,dvpar/dt]
    """
    Bb = self.B(r,phi,z)
    B = np.linalg.norm(Bb)
    Bg = self.gradAbsB(r,phi,z)
    b = B/B
    vperp_squared = self.FUSION_ALPHA_SPEED_SQUARED - vpar**2
    c = self.ALPHA_PARTICLE_MASS /self.ALPHA_PARTICLE_CHARGE/B/B/B
    mu = vperp_squared/2/B

    # compute d/dt (r,phi,z)
    dot_xyz = b*vpar +c*(vperp_squared/2 + vpar**2) * np.cross(Bb,Bg)
    J =self.jac_cart_to_cyl(r,phi,z)
    dot_rphiz = J @ dot_x

    # compute d/dt (vpar)
    dot_vpar = -(mu/vpar)*Bg @ dot_rphiz

    # compile into a vector
    dot_state =  np.append(dot_rphiz,dot_vpar)
    return dot_state
    


  def startup(self):
    """
    Evaluate u0 over the mesh.
    """
    # mesh spacing
    r_lin = np.arange(self.rmin,self.rmax,self.dr)
    phi_lin = np.arange(self.phimin,self.phimax,self.dphi)
    z_lin = np.arange(self.zmin,self.zmax,self.dz)
    vpar_lin = np.arange(self.vparmin,self.vparmax,self.dvpar)
    n_r = len(r_lin)
    n_phi = len(phi_lin)
    n_z = len(z_lin)
    n_vpar = len(vpar_lin)

    # build the grid
    r_grid,phi_grid,z_grid,vpar_grid = np.meshgrid(r_lin,phi_lin,
                             z_lin,vpar_lin,indexing='ij',sparse=True)
    ## evaluate 
    #U_grid = np.zeros_like(r_grid)
    #for ii in range(n_r):
    #  for jj in range(n_theta):
    #    for kk in range(n_phi):
    #      for ll in range(n_vpar):
    #        U_grid[ii,jj,kk,ll] = self.u0(r_grid[ii,jj,kk,ll],theta_grid[ii,jj,kk,ll],
    #                         phi_grid[ii,jj,kk,ll],vpar_grid[ii,jj,kk,ll])

    # reshape the grid to points
    X = np.vstack((np.ravel(self.r_grid),np.ravel(self.phi_grid),
            np.ravel(self.z_grid),np.ravel(self.vpar_grid))).T

    # compute initial density along the mesh, and reshape it
    U_grid = np.array([self.u0(*xx) for xx in X])
    U_grid = np.reshape(U_grid,np.shape(r_grid))

    # store the value of the density
    self.U_grid = np.copy(U_grid)

    # save the grids for interpolation
    self.r_grid = r_grid
    self.phi_grid = phi_grid
    self.z_grid = z_grid
    self.vpar_grid = vpar_grid
    return 

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
    
    # reshape the grid to points
    X = np.vstack((np.ravel(self.r_grid),np.ravel(self.phi_grid),
            np.ravel(self.z_grid),np.ravel(self.vpar_grid))).T
    # backwards integrate the points
    if self.integration_method == "euler":
      # TODO: verify correctness
      G =  np.array([self.GC_rhs(*xx) for xx in X])
      Xtm1 = X - self.dt*G
    elif self.integration_method == "midpoint":
      # TODO: verify correctness
      G =  np.array([self.GC_rhs(*xx) for xx in X])
      Xstar = np.copy(X - self.dt*G/2)
      G =  np.array([self.GC_rhs(*xx) for xx in Xstar])
      Xtm1 = X - self.dt*G/2

    elif self.integration_method == "rk4":
      # TODO: verify correctness
      G = np.array([self.GC_rhs(*xx) for xx in X])
      k1 = np.copy(dt*G)
      G = np.array([self.GC_rhs(*xx) for xx in X+k1/2])
      k2 = np.copy(dt*G)
      G = np.array([self.GC_rhs(*xx) for xx in X+k2/2])
      k3 = np.copy(dt*G)
      G = np.array([self.GC_rhs(*xx) for xx in X+k3])
      k4 = np.copy(dt*G)
      Xtm1 = np.copy(X -(k1+2*k2+2*k3+k4)/6)

    return Xtm1

  def interpolate(self,X):
    """
    Interpolate the value of u(r,theta,phi,vpar) from grid points to 
    the set of points X.

    input:
    X: (N,4) array, points at which to interpolate the value of u(x,y,z,vpar)

    return: 
    U: (N,) array, interpolated values of u(state) for state in X.
    """
    # TODO: should we use sparse arrays for U_grid to avoid computation
    # with zeros?
    interpolator = RegularGridInterpolator((self.r_grid,self.phi_grid,
                   self.z_grid,self.vpar_grid), self.U_grid)
    UX = interpolator(X)
    return UX

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

  def apply_boundary_conds(self,X):
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
    idx_up =  X_feas[:,3] > self.vparmax
    X_feas[:,3][idx_up] = self.vparmin + (X_feas[:,3][idx_up]-vparmax)%vpardiff
    idx_down =  X_feas[:,3] < self.vparmin
    X_feas[:,3][idx_up] = self.vparmax - (vparmin-X_feas[:,3][idx_down])%vpardiff

    return np.copy(X_feas),idx_feas
    

  def solve(self):
    """
    Perform the pde solve.
    """
    # build the mesh and evalute U
    self.startup()

    # only backwards integrate once b/c time independent
    X = self.backstep()

    # allocate space for density values
    UX = np.zeros(len(X))

    # collect departure points within the meshed volume
    X_feas,idx_feas = self.apply_boundary_conds(X)
    
    times = np.arange(self.tmin,self.tmax,self.dt)
    for tt in times:

      # intepolate the values of the departure points
      # this step assumes the dirichlet boundary conditions
      # i.e. that UX[not idx_feas] == 0
      UX[idx_feas] = self.interpolate(X_feas)

      # forward step: set U(grid,t+1) from UX
      self.update_U_grid(UX)

    return self.U_grid

  def compute_spatial_marginal(self):
    """ 
    Compute the marginal density over the spatial variables.
    """
    raise NotImplementedError

  def compute_loss_fraction(self,pbndry):
    """
    Compute the loss fraction by integrating the probability density
    over the plasma boundary.

    input: plasma boundary should be a radial function, r(theta,z) = pndry(phi,z).
    return: probability mass of particles within pbndry.
    """
    raise NotImplementedError
