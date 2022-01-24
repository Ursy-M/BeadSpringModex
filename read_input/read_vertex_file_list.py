
"""
Small function to read a vertex file list of a fiber
"""


import numpy as np

try:
    import read_vertex_file
except ImportError:
    from read_input import read_vertex_file


def read_vertex_file_list(name_files):
    """
    Parameters
    ----------
    name_files : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    ref_config : TYPE
        DESCRIPTION.

    """
    comment_symbols = ['#']
    ref_config = []
    with open(name_files, 'r') as f:
        for line in f:
            # strip comments
            if comment_symbols[0] in line:
                line, comment = line.split(comment_symbols[0], 1)
            
            # ignore blank lines
            line = line.strip()
            if line != '':
                config = read_vertex_file.read_vertex_file(line.split()[0])[1]
                ref_config.append(config)
    
    
    return len(ref_config)
    
                


                
    