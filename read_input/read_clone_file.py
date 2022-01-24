
"""
Small function to read a file with the initial locations  
"""


import numpy as np


def read_clone_file(name_file):
    """
    Parameters
    ----------
    name_file : string
        
    Returns
    -------
    - number_of_bodies (fibers or obstacles)
    - locations (Nbodies, 3)

    """
    
    comment_symbols = ['#']
    with open(name_file, 'r') as f:
        locations = []
        i = 0
        for line in  f:
            # strip comments
            if comment_symbols[0] in line:
                line, comment = line.split(comment_symbols[0], 1)
                
            # ignore blank lines
            line = line.strip()
            if line != '':
                if i == 0:
                    number_of_bodies = int(line.split()[0])
                else :
                    data = line.split()
                    location = [float(data[0]), float(data[1]), float(data[2])]
                    locations.append(location)
                i += 1
                if i == number_of_bodies + 1:
                    break
        
        # create and return numpy arrays
        locations = np.array(locations)
        
        return number_of_bodies, locations            