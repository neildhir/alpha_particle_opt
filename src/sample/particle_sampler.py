import numpy as np
from src.sample.radial_density import RadialDensity
from src.utils.constants import V_MAX  
from simsopt.util.mpi import MpiPartition

class ParticleSampler:

    def __init__(self, nfp, s_label=0.25, n_particles = 128):
        """
        A class for randomly sampling or positioning particles in flux coordinates.

        nfp: number of field periods.
        s_label: str, the normalized toroidal flux (0, 1) for the surface that will
            be sampled by the sample_surface function.
        n_particles: int, number of particles to sample, if using the random sampling method.
        """

        self.s_label = s_label
        self.nfp = nfp
        self.n_particles = n_particles

        self.mpi: MpiPartition = MpiPartition(1)

    def sample_surface(self):
        """
        Nonuniformly sample a flux surface with normalized toroidal flux s_label. 
        The toroidal and poloidal values are sampled uniformly on [0, 2pi/nfp] x [0, 2pi].
        The parallel velocity is sampled uniformly over [-V_MAX, V_MAX].

        return stz_inits, vpar_inits
        stz_inits: (n_particles, 3) array of points (s, theta, zeta) on the flux surface.
        vpar_inits: (n_particles,) array of parallel velocity values
        """
        n_particles = self.n_particles

        # sampling over (theta,zeta,vpar) for a fixed surface
        s_inits = self.s_label*np.zeros(n_particles)
        theta_inits = np.zeros(n_particles)
        zeta_inits = np.zeros(n_particles)
        vpar_inits = np.zeros(n_particles)

        if self.mpi.proc0_groups:
            # randomly sample theta,zeta,vpar
            theta_inits = np.random.uniform(0, 2 * np.pi, n_particles)
            zeta_inits = np.random.uniform(0, 2 * np.pi / self.nfp, n_particles)
            vpar_inits = np.random.uniform(-V_MAX, V_MAX, n_particles)
        
        # broadcast the points
        self.mpi.comm_groups.Bcast(theta_inits, root=0)
        self.mpi.comm_groups.Bcast(zeta_inits, root=0)
        self.mpi.comm_groups.Bcast(vpar_inits, root=0)

        # stack the samples
        stz_inits = np.vstack((s_inits, theta_inits, zeta_inits)).T
        return stz_inits, vpar_inits

    def sample_volume(self):
        """
        Sample the stellarator volume. 
        The angles and the parallel velocity are sampled by sample_surface.
        The normalized toroidal flux is sampled using the radial density sampler.

        return stz_inits, vpar_inits
        stz_inits: (n_particles, 3) array of points (s, theta, zeta).
        vpar_inits: (n_particles,) array of parallel velocity values
        """            
        # randomly sample the angles and parallel velocity
        stz_inits, vpar_inits = self.sample_surface()
        n_particles = len(vpar_inits)

        # sample s
        s_inits = np.zeros(n_particles)
        if self.mpi.proc0_groups:
            sampler = RadialDensity()
            s_inits = sampler.sample(n_particles)

        self.mpi.comm_groups.Bcast(s_inits, root=0)

        stz_inits[:,0] = s_inits.flatten()

        return stz_inits, vpar_inits

    def sample_surface_uniformly(self):
        # TODO: implement uniform surface sampling.
        raise NotImplementedError


class FluxSurfaceGrid:

    def __init__(self,
                 s_label=0.25,
                 nfp=4, 
                 ntheta=32, 
                 nzeta=32, 
                 nvpar=32):
        """
        A class for computing tensor product grids on flux surfaces.

        nfp: number of field periods.
        s_label: float or str, either the normalized toroidal flux (0, 1) or the 
            string 'volume' denoting the volume it is sampled in.
        ntheta, nzeta, nvpar: number of points taken per direction, if a grid is used.

        """
        self.s_label = s_label
        self.nfp = nfp
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.nvpar = nvpar

    def surface_grid(self):
        """
        Build a tensor product grid on a single surface in flux coordinates. The grid is
        uniformly spaced over the two angles and parallel velocity.

        return stz_inits, vpar_inits
        stz_inits: (ntheta*nzeta*nvpar, 3) array of points (s, theta, zeta) on the grid.
        vpar_inits: (n_vpar,) array of parallel velocity values
        """
        theta1d = np.linspace(0, 2 * np.pi, self.ntheta, endpoint=False)
        zeta1d = np.linspace(0, 2 * np.pi / self.nfp, self.nzeta, endpoint=False)
        vpar1d = np.linspace(-V_MAX, V_MAX, self.nvpar)

        # build a mesh
        [thetas, zetas, vpars] = np.meshgrid(theta1d, zeta1d, vpar1d)
        stz_inits = np.zeros((self.ntheta * self.nzeta * self.nvpar, 3))
        stz_inits[:, 0] = self.s_label
        stz_inits[:, 1] = thetas.flatten()
        stz_inits[:, 2] = zetas.flatten()
        vpar_inits = vpars.flatten()
        return stz_inits, vpar_inits
    

if __name__ == "__main__":
    sampler = ParticleSampler(nfp=4, s_label=0.25, n_particles=2)
    print(sampler.sample_surface())
    print(sampler.sample_volume())

    sampler = FluxSurfaceGrid(nfp=4, s_label=0.3, ntheta=1, nzeta=3, nvpar=2)
    print(sampler.surface_grid())
