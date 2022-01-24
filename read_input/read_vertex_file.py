
"""
Small function to read a vertex file of a body (fiber or obstacle)
"""

import numpy as np

def read_vertex_file(name_file):
    """
    Parameters
    ----------
    name_file : string

    Returns
    -------
    - number of beads
    - coordinates of the fiber or obstacle beads (number_of_beads, 3)

    """
    
    comment_symbols = ['#']
    coordinates = []
    with open(name_file, 'r') as f:
        i = 0
        for line in f:
            # strip comments
            if comment_symbols[0] in line:
                line, comment = line.split(comment_symbols[0], 1)
            
            # ignore blank lines
            line = line.strip()
            if line != '':
                if i ==0:
                    number_of_beads = int(line.split()[0])
                else:
                    location = np.fromstring(line, sep = ' ')
                    coordinates.append(location)
                i += 1
    
    coordinates = np.array(coordinates)
    
    return number_of_beads, coordinates                    
    