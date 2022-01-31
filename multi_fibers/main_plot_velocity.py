
"""
main file to plot velocity field
"""

import sys
import os
import argparse
import numpy as np
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
        from mobility import mobility as mob
        
        found_functions = True
    except ImportError as exc:
        sys.stderr.write('Error: failed to import settings module ({})\n'.format(exc))
        path_to_append += '../'
        print('searching functions in path ', path_to_append)
        sys.path.append(path_to_append)
        if len(path_to_append) > 21:
            print('\nProject functions not found. Edit path in main_plot_velocity.py or check PYTHONPATH')
            sys.exit()

# try to import the visit_writer (boost implementation)
try:
    import visit.visit_writer as visit_writer
except ImportError:
    print('Failed to import visit_writer')
    pass

# =============================================================================
# define modules
# =============================================================================
def set_grid_coordinates(grid, *args, **kwargs):
    """
    Parameters
    ----------
    grid : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    grid_coordinates : TYPE
        DESCRIPTION.

    """
    # prepare grid values
    grid = np.reshape(grid, (3,3)).T
    grid_length = grid[1] - grid[0]
    grid_points = np.array(grid[2], dtype=np.int32)
    num_points = grid_points[0] * grid_points[1] * grid_points[2]
    
    # set grid coordinates
    delta = grid_length / grid_points
    grid_x = np.array([grid[0,0] + delta[0] * (x + 0.5) for x in range(grid_points[0])])
    grid_y = np.array([grid[0,1] + delta[1] * (x + 0.5) for x in range(grid_points[1])])
    grid_z = np.array([grid[0,2] + delta[2] * (x + 0.5) for x in range(grid_points[2])])
    
    zz, yy, xx = np.meshgrid(grid_z, grid_y, grid_x, indexing='ij')
    grid_coordinates = np.zeros((num_points, 3))
    grid_coordinates[:,0] = np.reshape(xx, xx.size)
    grid_coordinates[:,1] = np.reshape(yy, yy.size)
    grid_coordinates[:,2] = np.reshape(zz, zz.size)
    
    return [grid_x, grid_y, grid_z], delta, grid_points, grid_coordinates


def plot_velocity_field(grid_coordinates, r_vectors, force, a_bead_fib, tracer_radius, eta, *args, **kwargs):
    
    
    # set bead (= a_bead_fib) and node (= 0) radii
    radius_source = np.ones(r_vectors.size // 3) * a_bead_fib
    radius_target = np.ones(grid_coordinates. size // 2) * tracer_radius
    
    # compute the velocity field
    domain = kwargs.get('domain')
    mobility_vector_prod_implementation = kwargs.get('mobility_vector_prod_implementation')
    if domain == 'single_wall':
        if mobility_vector_prod_implementation.find('pycuda') > - 1:
            grid_velocity = mob.single_wall_mobility_trans_times_force_source_target_pycuda(r_vectors,
                                                                                            grid_coordinates,
                                                                                            force,
                                                                                            radius_source,
                                                                                            radius_target,
                                                                                            eta,
                                                                                            *args,
                                                                                            **kwargs)
        elif mobility_vector_prod_implementation.find('numba') > -1:
            grid_velocity = mob.single_wall_mobility_trans_times_force_source_target_numba(r_vectors,
                                                                                           grid_coordinates,
                                                                                           force,
                                                                                           radius_source,
                                                                                           radius_target,
                                                                                           eta,
                                                                                           *args,
                                                                                           **kwargs)
        else:
            grid_velocity = mob.single_wall_mobility_trans_times_force_source_target(r_vectors,
                                                                                     grid_coordinates,
                                                                                     force,
                                                                                     radius_source,
                                                                                     radius_target,
                                                                                     eta,
                                                                                     *args,
                                                                                     **kwargs)
    else:
        if mobility_vector_prod_implementation.find('pycuda') > - 1:
            grid_velocity = mob.no_wall_mobility_trans_times_force_source_target_pycuda(r_vectors,
                                                                                        grid_coordinates,
                                                                                        force,
                                                                                        radius_source,
                                                                                        radius_target,
                                                                                        eta,
                                                                                        *args,
                                                                                        **kwargs)
        elif mobility_vector_prod_implementation.find('numba') > -1:
            grid_velocity = mob.no_wall_mobility_trans_times_force_source_target_numba(r_vectors,
                                                                                       grid_coordinates,
                                                                                       force,
                                                                                       radius_source,
                                                                                       radius_target,
                                                                                       eta,
                                                                                       *args,
                                                                                       **kwargs)
        else:
            grid_velocity = mob.no_wall_mobility_trans_times_force_source_target(r_vectors,
                                                                                 grid_coordinates,
                                                                                 force,
                                                                                 radius_source,
                                                                                 radius_target,
                                                                                 eta,
                                                                                 *args,
                                                                                 **kwargs)
    
    
    return grid_velocity


# =============================================================================
# main
# =============================================================================
# get command line
parser = argparse.ArgumentParser(description='run a simulation to plot velocity.')
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
n_steps = n_steps_per_unit_time * n_settling_time
a_bead_fib = read.fiber_bead_radius
tracer_radius = read.tracer_radius
eta = read.eta
mobility_vector_prod_implementation = read.mobility_beads_implementation
domain = read.domain
output_name = read.output_name
generate_vtk_files = read.generate_vtk_files
grid = read.grid

# set grid nodes 
[grid_x, grid_y, grid_z], delta, grid_points, grid_coordinates = set_grid_coordinates(grid)


# load ouput data (positions and forces)
path = 'output'     
current_directory = os.getcwd()
directory = current_directory + '/' + path + '/'
directory_grid_velocity = directory + 'output_grid_velocity' + '/'
positions = np.loadtxt(directory + 'run.output_positions.csv', delimiter=';')
forces = np.loadtxt(directory + 'run.output_forces.csv', delimiter=';')

# get the number of beads
number_of_beads = positions.shape[1] // 3

# loop over time steps
start_time = datetime.now
for step in range(n_steps):
    if step % n_save == 0 and step >=0:
        elapsed_time = datetime.now - start_time
        print('number of steps = ', step)
        print('step = ', step, ', wallclock = ', elapsed_time)
        
        all_position_x = []
        all_position_y = []
        all_position_z = []
        
        all_force_x    = []
        all_force_y    = []
        all_force_z    = []
        
        for k in range(number_of_beads):
            all_position_x.append(positions[:,(3*k)][step])
            all_position_y.append(positions[:,(3*k+1)][step])
            all_position_z.append(positions[:,3*k+2][step])
            
            all_force_x.append(forces[:,(3*k)][step])
            all_force_y.append(forces[:,(3*k+1)][step])
            all_force_z.append(forces[:,3*k+2][step])
            
        # get the corresponding positions
        r_vectors = np.array([[all_position_x[i], all_position_y[i], all_position_z[i]] for i in range(number_of_beads)])
        
        # get the corresponding forces
        force = np.array([[all_force_x[i], all_force_y[i], all_force_z[i]] for i in range(number_of_beads)])
        
        # compute grid velocity
        grid_velocity = plot_velocity_field(grid_coordinates, 
                                            r_vectors, 
                                            force, 
                                            a_bead_fib, 
                                            tracer_radius, 
                                            eta,
                                            mobility_vector_prod_implementation,
                                            domain)
            
        if generate_vtk_files == 'True':
            # prepare data for VTK writer
            variables  = [np.reshape(grid_velocity, grid_velocity.size)]
            dims = np.array([grid_points[0]+1, grid_points[1]+1, grid_points[2]+1], dtype = np.int32)
            nvars = 1
            vardims = np.array([3])
            centering = np.array([0])
            varnames = ['velocity\0']
            name = directory_grid_velocity + output_name + '.velocity_field_' + str(step) + '.vtk'
            grid_x = grid_x - delta[0] * 0.5
            grid_y = grid_y - delta[1] * 0.5
            grid_z = grid_z - delta[2] * 0.5
            grid_x = np.concatenate([grid_x, [grid[1,0]]])
            grid_y = np.concatenate([grid_y, [grid[1,1]]])
            grid_z = np.concatenate([grid_z, [grid[1,2]]])
            
            # write velocity field
            visit_writer.boost_write_rectilinear_mesh(name,         # file's name
                                                      0,            # 0=ASCII, 1 = Binary
                                                      dims,         # {mx, my, mz}
                                                      grid_x,       # xmesh
                                                      grid_y,       # ymesh
                                                      grid_z,       # zmesh
                                                      nvars,        # number of variables
                                                      vardims,      # size of each varibles, 1=scalar, velocity=3*scalars
                                                      centering,    # write to cell centers of corners 
                                                      varnames,     # variable's names
                                                      variables)    # variables
            
        else:
            # write velocity field
            name = directory_grid_velocity + output_name + '.velocity_field_' + str(step) + '.csv'
            np.savetxt(name, grid_velocity, delimiter=';')
            
            
duration = datetime.now() - start_time
print('time :\t' + 'execution\t' + str(duration))            