
"""
Small class to read an input file to run a simulation.
"""

import numpy as np
import ntpath



class ReadInput(object):
    
    def __init__(self, entries):
        self.input_file = entries
        self.options = {}
        number_kind_of_fibers = 0
        
        # read input file
        comment_symbols = ['#']
        with open(self.input_file, 'r') as f:
            # loop over lines
            for line in f:
                # strip comments
                if comment_symbols[0] in line:
                    line, comment = line.split(comment_symbols[0], 1)
                
                # save options to dictionary, value may be more than one world
                line = line.strip()
                if line != '':
                    option, value = line.split(None, 1)
                    if option == 'fiber':
                        option += str(number_kind_of_fibers)
                        number_kind_of_fibers += 1
                    self.options[option] = value
                
        
        # set option to file or default values
        self.n_settling_time = int(self.options.get('n_settling_time') or 1)
        self.n_steps_per_unit_time = int(self.options.get('n_steps_per_unit_time') or 30)
        self.initial_step = int(self.options.get('initial_step') or 0)
        self.n_save = int(self.options.get('n_save') or 1)
        self.dt = float(self.options.get('dt') or 0.0)
        self.eta = float(self.options.get('eta') or 1.0)
        self.fiber_bead_radius = float(self.options.get('fiber_bead_radius') or 1.0)
        self.centerline_distance_factor = float(self.options.get('centerline_distance_factor') or 2.2)
        self.tracer_radius = float(self.options.get('tracer_radius') or 0.0)
        self.weight_per_unit_length = float(self.options.get('weight_per_unit_length') or 1.0)
        self.elasto_gravitational_number = float(self.options.get('elasto_gravitational_number') or 100.0)
        self.mobility_beads_implementation = str(self.options.get('mobility_beads_implementation') or 'python')
        self.mobility_vector_prod_implementation = str(self.options.get('mobility_vector_prod_implementation') or 'python')
        self.output_name = str(self.options.get('output_name') or 'output')
        self.save_velocities = str(self.options.get('save_velocities') or 'False')
        self.save_forces = str(self.options.get('save_forces') or 'False')
        self.domain = str(self.options.get('domain') or 'no_wall')
        self.repulsion_strength = float(self.options.get('repulsion_strength') or 1.0)
        self.stiffness_paramater = float(self.options.get('stiffness_paramater') or 2.0)
        self.contact_distance_factor = float(self.options.get('contact_distance_factor') or 1.1)
        self.set_steric_forces_implementation = str(self.options.get('set_steric_forces_implementation') or 'python')
        self.periodic_length = np.fromstring(self.options.get('periodic_length') or '0 0 0', sep=' ')
        self.scheme = str(self.options.get('scheme') or 'BDF')
        self.generate_vtk_files = str(self.options.get('generate_vtk_files') or 'False')
        self.grid = np.fromstring(self.options.get('grid') or 'None', sep=' ')
        
        # create list with [vertex_file, clones_file]
        self.number_kind_of_free_bodies = number_kind_of_fibers
        self.fibers = []
        self.fibers_ID = []
        
        # create list with [vertex_file, clones_file] for each kind of fibers
        for i in range(number_kind_of_fibers):
            option = 'fiber' + str(i)
            fiber_files = str.split(str(self.options.get(option)))
            self.fibers.append(fiber_files)
        
        # create ID for each kind of fibers
        for fib in self.fibers:
            # first remove directory from fiber name
            head, tail = ntpath.split(fib[1])
            # remove end (.clones)
            tail = tail[:-7]
            self.fibers_ID.append(tail)
        
        return
        
        