# BeadSpring model for flexible fibers in Stokes flow

This package contains python codes to run a simulation of several fibers at low Reynolds number. The fiber is
discretized as a serie of spherical beads connected by spring constraints. The motion of each bead in Stokes flow is
given by the mobility relation, which relates the bead velocities to external forces applied on each bead.

This package is inspired from the [RigidMultiblobsWall](https://github.com/stochasticHydroTools/RigidMultiblobsWall) organization.

### References
For the theory behind, see:
[1]
[2]
[3]

### Organization
* **configuration/**: it contains generated .vertex and .clones files to run examples.
* **fiber/**: it contains a class to handle a single fiber.
* **force/**: it contains functions to compute external and internal elastic forces, see [1] and [3]. 
* **generate_vertex_and_clone_files/**: it contains main files to generate .vertex and .clones files.
* **integrator/**: it contains the solver and the scheme to integrate the equation of motion.
* **mobility/**: it contains functions to compute the mobility matrix **M** and the product **MF** respectively via
* CPU(accelerated with numba) and GPU, see [2] and [3].
* **multi_fibers/**: it contains main files to run multi-fibers simulations, currently adapted in the case of
* sedimentation at low Reynolds number.
* **quaternion/**: it contains the class to handle quaternions.
* **read_input/**: it contains the class to read input data.
* **visit/**:

### Usage



### Contact
Please contact Ursy Makanga (ursy.makanga@polytechnique.edu) for any problems or suggestions regarding this package. 



