
"""
main file to run a simultion with n fibers
"""
import sys
import numpy as np
import argparse
from scipy.integrate import ode
from functools import partial
from datetime import datetime


# =============================================================================
# import project functions
# =============================================================================
# find project functions
found_functions = False
path_to_append = ''
while found_functions is False:
    try:    
        from read_input import read_input
        from read_input import read_clone_file
        from read_input import read_vertex_file_list
        from fiber import fiber
        from integrator.integrator import Integrator
        from force import force

        found_functions = True
    except ImportError as exc:
        sys.stderr.write('Error: failed to import settings module ({})\n'.format(exc))
        path_to_append += '../'
        print('searching functions in path ', path_to_append)
        sys.path.append(path_to_append)
        if len(path_to_append) > 21:
            print('\nProject functions not found. Edit path in main_multi_fibers.py or check PYTHONPATH')
            sys.exit()
        
# =============================================================================
# define modules
# =============================================================================
def get_beads_r_vectors(bodies_fibers, number_of_beads):
    """
    Parameters
    ----------
    bodies_fibers : TYPE
        DESCRIPTION.
    number_of_beads : TYPE
        DESCRIPTION.

    Returns
    -------
    r_vectors : TYPE
        DESCRIPTION.

    """
    r_vectors = np.empty((number_of_beads, 3))
    offset = 0
    for x in bodies_fibers:
        NF = x.NF
        r_vectors[offset:(offset + NF)] = x.get_fiber_r_vectors()       
        offset += NF
    
    return r_vectors


def get_beads_list_indices(bodies_fibers):
    """
    Parameters
    ----------
    bodies_fibers : TYPE
        DESCRIPTION.
        
    Returns
    -------
    bead_list_index : TYPE
        DESCRIPTION.

    """
    bead_list_index = []
    for k, f in enumerate(bodies_fibers):
        NF = f.NF
        bead_list_index.append(k*np.ones(NF, dtype=int))
    
    
    bead_list_index = [item for sublist in bead_list_index for item in sublist]
    
    return bead_list_index


# =============================================================================
# main
# =============================================================================
# get command line
parser = argparse.ArgumentParser(description='run a multi_fibers simulation and save trajectory.')
parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', help='name of the input file')
args = parser.parse_args()
input_file = args.input_file

# read the input file
read = read_input.ReadInput(input_file)

# set some useful entries for the simualtion
n_settling_time = read.n_settling_time
initial_step = read.initial_step
n_steps_per_unit_time = read.n_steps_per_unit_time
n_save = read.n_save
dt = read.dt
eta = read.eta
W = read.weight_per_unit_length
B = read.elasto_gravitational_number
a_bead_fib = read.fiber_bead_radius
DL_factor = read.centerline_distance_factor
output_name = read.output_name
fibers = read.fibers
fibers_ID = read.fibers_ID

# set some implementations
force.calc_steric_forces = force.set_steric_forces(read.set_steric_forces_implementation)


body_types = []
body_names = []
bodies_fibers = []
# create fibers 
for ID, fib in enumerate(fibers):
    print('creating fibers =', fib[1])
    # read vertex and clones files
    number_of_beads_in_fiber = read_vertex_file_list.read_vertex_file_list(fib[0])   
    number_of_beads_all, fiber_bead_locations = read_clone_file.read_clone_file(fib[1])
    body_types.append(number_of_beads_all // number_of_beads_in_fiber)
    body_names.append(fibers_ID[ID])
    # create each body of type fiber
    for i in range(number_of_beads_all // number_of_beads_in_fiber):
        f = fiber.Fiber(fiber_bead_locations[i*number_of_beads_in_fiber:(i+1)*number_of_beads_in_fiber,:],
                        DL_factor,
                        a_bead_fib)
        f.ID = fibers_ID[ID]
        # append fiber to the total bodies list
        bodies_fibers.append(f)

        
# set some variables
number_types_fiber = len(body_types)
bodies_fibers = np.array(bodies_fibers)
number_bodies_fiber = len(bodies_fibers)
number_of_beads_fiber = sum(x.NF for x in bodies_fibers)
number_of_beads = int(number_of_beads_fiber)
radius_array = np.ones((number_of_beads))
radius_array[0:number_of_beads_fiber] *= a_bead_fib
bead_list_index = get_beads_list_indices(bodies_fibers) 
# initial values        
r_vectors = get_beads_r_vectors(bodies_fibers, number_of_beads)
initial_values = np.squeeze(np.asarray(r_vectors.reshape(1, 3*number_of_beads)))

# save system information
with open(output_name + '.system_info', 'w') as g:
    g.write('number_of_fiber_types     ' + str(number_types_fiber) + '\n')
    g.write('body_names                ' + str(body_names) + '\n')
    g.write('number_of_fibers          ' + str(number_bodies_fiber) + '\n')
    g.write('number_of_beads           ' + str(number_of_beads) + '\n')

g.close()


# create integrator
integrator = Integrator(bodies_fibers, 
                        number_of_beads, 
                        scheme=read.scheme, 
                        domain=read.domain,
                        mobility_vector_prod_implementation=read.mobility_vector_prod_implementation)

integrator.calc_gravitational_forces     = partial(force.calc_gravitational_forces,
                                                   weight_per_unit_length=W)

integrator.calc_steric_forces            = partial(force.calc_steric_forces,
                                                   periodic_length=read.periodic_length,
                                                   repulsion_strength=read.repulsion_strength,
                                                   stiffness_parameter=read.stiffness_paramater,
                                                   contact_distance_factor=read.contact_distance_factor,
                                                   radius_array=radius_array,
                                                   bead_list_index=bead_list_index)

integrator.calc_stretching_forces        = partial(force.calc_stretching_forces,
                                                   weight_per_unit_length=W,
                                                   elasto_gravitational_number=B)           

integrator.calc_bending_forces           = partial(force.calc_bending_forces,
                                                   weight_per_unit_length=W,
                                                   elasto_gravitational_number=B)

integrator.get_beads_r_vectors           = get_beads_r_vectors
integrator.eta                           = eta
integrator.a_bead_fib                    = a_bead_fib
integrator.periodic_length               = read.periodic_length
integrator.weight_per_unit_length        = W
integrator.number_of_beads_fiber         = number_of_beads_fiber


# ode solver
final_step = n_settling_time * n_steps_per_unit_time * dt
t = np.arange(initial_step, final_step, dt)                                
r = ode(integrator)
r.set_integrator('vode', method=read.scheme, with_jacobian=True)
r.set_initial_value(initial_values, initial_step)
sol = np.zeros((len(t), len(initial_values)))
sol_velocities = np.zeros((len(t), len(initial_values)))
all_t = np.zeros((len(t), 1))
idx = 0

# loop over time steps
start_time = datetime.now()
while r.successful() and r.t < t[-1]:
    print('integrator = ', read.scheme, ', step = ', idx,  ', t + dt = ', r.t + dt)
    if idx % n_save == 0 and idx >= 0 :
        sol[idx, :] = r.y 
        all_t[idx, :] = t[idx]
        if read.save_velocities == 'True':
            sol_velocities[idx, :] = integrator.velocities
    r.integrate(r.t + dt)
    idx += 1


# save
np.savetxt(output_name + ".output_positions.csv", sol, delimiter=';')
np.savetxt(output_name + ".output_times.csv", all_t, delimiter=';')
if read.save_velocities == 'True':
    np.savetxt(output_name + ".output_velocities.csv", sol_velocities, delimiter=';')
    
    
duration = datetime.now() - start_time
print('time :\t' + 'execution\t' + str(duration))
print('\n\n\n# End')
       

