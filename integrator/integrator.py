
"""
Integrator for multifibers simulation.
"""


import numpy as np
from mobility import mobility as mob


class Integrator(object):
    
    def __init__(self, bodies_fibers, number_of_beads, scheme, domain, mobility_vector_prod_implementation='numba'):
        """
        init object
        """
        self.bodies_fibers = bodies_fibers
        self.number_of_beads = number_of_beads
        self.scheme = scheme
        self.domain = domain
        
        # other entries
        self.get_beads_r_vectors = None
        self.eta = None
        self.a_bead_fib = None
        self.weight_per_unit_length = None
        self.velocities = np.zeros((1, 3*self.number_of_beads))
        self.forces = np.zeros((1, 3*self.number_of_beads))
        self.periodic_length = None
        self.number_of_beads_fiber = None
        
        
        if self.domain == 'single_wall':
            if mobility_vector_prod_implementation.find('pycuda') > -1:
                self.mobility_trans_times_force = mob.single_wall_mobility_trans_times_force_pycuda
            elif mobility_vector_prod_implementation.find('numba') > -1:
                self.mobility_trans_times_force = mob.single_wall_mobility_trans_times_force_numba
        elif self.domain == 'no_wall':
            if mobility_vector_prod_implementation.find('pycuda') > -1:
                self.mobility_trans_times_force = mob.no_wall_mobility_trans_times_force_pycuda
            elif mobility_vector_prod_implementation.find('numba') > -1:
                self.mobility_trans_times_force = mob.no_wall_mobility_trans_times_force_numba
        elif self.domain == 'in_plane':
            if mobility_vector_prod_implementation.find('pycuda') > -1:
                self.mobility_trans_times_force = mob.in_plane_mobility_trans_times_force_pycuda
            elif mobility_vector_prod_implementation.find('numba') > -1:
                self.mobility_trans_times_force = mob.in_plane_mobility_trans_times_force_numba
        
        
    def mobility_matrix(self, force, r_vectors, eta, a_bead_fib, periodic_length):
        """
        Parameters
        ----------
        force : TYPE
            DESCRIPTION.
        r_vectors : TYPE
            DESCRIPTION.
        eta : TYPE
            DESCRIPTION.
        a_bead_fib : TYPE
            DESCRIPTION.
        periodic_length : TYPE
            DESCRIPTION.

        Returns
        -------
        velocity : TYPE
            DESCRIPTION.

        """
        # compute M^tt \times F
        velocity = self.mobility_trans_times_force(r_vectors, force, eta, a_bead_fib, periodic_length=periodic_length)
        
        return velocity
        
        
    def solve_mobility_problem(self, *args, **kwargs):
        """
        solve the mobility problem
        compute the velocities on the beads subject to external forces.
        
        the linear velocities are sorted like (v1, v2,...)

        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # get beads coordinates
        r_vectors = self.get_beads_r_vectors(self.bodies_fibers, self.number_of_beads)
        
        # compute force on beads
        # compute gravitational  forces
        force = self.calc_gravitational_forces(r_vectors, 
                                               self.bodies_fibers)   
        # compute steric interactions
        force += self.calc_steric_forces(r_vectors)                     
        
        # compute bending foces
        force += self.calc_bending_forces(r_vectors,
                                          self.bodies_fibers)
        
        # compute stretching forces
        force += self.calc_stretching_forces(r_vectors,
                                              self.bodies_fibers)               
        
        force = np.reshape(force, force.size)
        
        # update forces
        self.forces = force
        
        # compute velocities --> M*F
        sol = self.mobility_matrix(force,
                                   r_vectors,
                                   self.eta,
                                   self.a_bead_fib,
                                   self.periodic_length)
            
        return sol
    
    
    def __call__(self, t, r_vectors):
        
        r_vectors = np.reshape(r_vectors, (self.number_of_beads, 3))
        offset = 0
        for x in self.bodies_fibers:
            offset_n = offset + x.NF
            x.fiber_bead_locations = r_vectors[offset:offset_n,:]        
            offset += x.NF
        
        
        # solve the mobility problem
        sol = self.solve_mobility_problem()
        
        
        self.velocities = np.squeeze(np.asarray(sol.reshape(1, 3*self.number_of_beads)))
        
        return self.velocities
        
        
             
                  
        
        
        
    
    
    