"""
Fluid mobility with pycuda (from usabiaga et al(2016))
"""


import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray



#precision = 'single'
precision = 'double'

mod = SourceModule("""
                   
#include <stdio.h>
#include <math.h>

//typedef float real;
typedef double real;                   

/*
    from P. J. Zuk, E. Wajnryb, K. A. Mizerski and P. Szymczak's paper.
    This function computes the translational-translational RPY tensors for same
    bead sizes.
*/
                   

__device__ void mobilityNW_kernel(real rx,
                                  real ry,
                                  real rz,
                                  real &Mxx,
                                  real &Mxy,
                                  real &Mxz,
                                  real &Myy,
                                  real &Myz,
                                  real &Mzz,
                                  const real inv_a,
                                  const int i,
                                  const int j)                   

{
 
real four_over_three = real(4.0) / real(3.0);

if (i == j){
        
    Mxx = four_over_three;
    Mxy = 0;
    Mxz = 0;
    Myy = Mxx;
    Myz = 0;
    Mzz = Mxx;

} 
else{
     
    rx = rx * inv_a;
    ry = ry * inv_a;
    rz = rz * inv_a;
    
    real r_square = rx*rx + ry*ry + rz*rz;
    real r = sqrt(r_square);
    
    real inv_r = real(1.0) / r;
    real inv_r_square = inv_r * inv_r;
    real c_1, c_2;

       if (r > 2){
           
            c_1 = real(1.0) + real(2.0) / (real(3.0) * r_square);
            c_2 = (real(1.0) - real(2.0) * inv_r_square) * inv_r_square;
            
            Mxx = (c_1 + c_2 * rx * rx) * inv_r;
            Mxy = (      c_2 * rx * ry) * inv_r;
            Mxz = (      c_2 * rx * rz) * inv_r;
            Myy = (c_1 + c_2 * ry * ry) * inv_r;
            Myz = (      c_2 * ry * rz) * inv_r;
            Mzz = (c_1 + c_2 * rz * rz) * inv_r;
        }
       else{
           
            c_1 = four_over_three * (real(1.0) - real(0.28125) * r);  // 9/32 = 0.28125
            c_2 = four_over_three * real(0.09375) * inv_r;            // 3/32 = 0.09375  
            
            Mxx = (c_1 + c_2 * rx * rx);
            Mxy = (      c_2 * rx * ry);
            Mxz = (      c_2 * rx * rz);
            Myy = (c_1 + c_2 * ry * ry);
            Myz = (      c_2 * ry * rz);
            Mzz = (c_1 + c_2 * rz * rz);
        }     
 }  
return;
 }                   

/*
    This function computes the product M*F to get velocities for
    the same bead sizes.
*/
                   
__global__ void velocity_function_no_wall_kernel(const real *x,
                                                 const real *f,
                                                 real *u,
                                                 const real eta,
                                                 const real a,
                                                 const int number_of_beads,
                                                 real Lx,
                                                 real Ly,
                                                 real Lz)

{
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i > number_of_beads) return;

real inv_a = real(1.0) / a;

real rx, ry, rz;
real Mxx, Mxy, Mxz;
real Myx, Myy, Myz;
real Mzx, Mzy, Mzz;
real Ux = 0;
real Uy = 0;
real Uz = 0;

int NDIM = 3;   // the spatial dimension
int ioffset = i * NDIM;
int joffset;

// Determine if the space is pseudo-periodic in any dimension
// We use a extended unit cell of length L = 3*(Lx, Ly, Lz)
int periodic_x = 0, periodic_y = 0, periodic_z = 0;

if (Lx > 0){
        periodic_x = 1;     
}
if (Ly > 0){
        periodic_y = 1;        
}
if (Lz > 0){
        periodic_z = 1;        
}

// loop over image boxes and then over beads
for (int boxX = -periodic_x; boxX <= periodic_x; boxX++){
        for (int boxY = -periodic_y; boxY <= periodic_y; boxY++){
                for (int boxZ = -periodic_z; boxZ <= periodic_z; boxZ++){        
                        for(int j = 0; j < number_of_beads; j++ ){
                                joffset = j * NDIM;
        
                                // compute the position vector between bead i and bead j
                                rx = x[ioffset    ] - x[joffset    ];
                                ry = x[ioffset + 1] - x[joffset + 1];
                                rz = x[ioffset + 2] - x[joffset + 2];
                                
                                // project a vector r to the extended unit cell
                                // centered around (0, 0, 0) and of size L = 3*(Lx, Ly, Lz)
                                // if any dimension of l is equal or smaller than zero
                                // the box is assumed to be infinite in that direction
                                if (Lx > 0){
                                    rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
                                    rx = rx + boxX * Lx;
                                }
                                if (Ly > 0){
                                    ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
                                    ry = ry + boxY * Ly;
                                }
                                if (Lz > 0){
                                    rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
                                    rz = rz + boxZ * Lz;
                                }
                                
                                // compute mobility of pair i-j, if i == j use self-interaction
                                int j_image = j;
                                if(boxX!=0 or boxY!=0 or boxZ!=0){
                                    j_image = -1;        
                                    }
                                mobilityNW_kernel(rx, ry, rz, Mxx, Mxy, Mxz, Myy, Myz, Mzz, inv_a, i, j_image);
                                Myx = Mxy;
                                Mzx = Mxz;
                                Mzy = Myz;
                                
                                // compute the product M_ij * F_j
                                Ux = Ux + (Mxx * f[joffset ] + Mxy * f[joffset + 1] + Mxz * f[joffset + 2]);
                                Uy = Uy + (Myx * f[joffset ] + Myy * f[joffset + 1] + Myz * f[joffset + 2]);
                                Uz = Uz + (Mzx * f[joffset ] + Mzy * f[joffset + 1] + Mzz * f[joffset + 2]);
                        } 
                }  
        }
}
// loop end        

// save velocity u_i
real pi = real(4.0) * atan(real(1.0));
real norm_fact_f = real(1.0) / (8 * pi * eta * a);

u[ioffset     ] = Ux * norm_fact_f;
u[ioffset +  1] = Uy * norm_fact_f;
u[ioffset +  2] = Uz * norm_fact_f;

return;
}
                  
""")


def real(x):
    if precision == 'single':
        return np.float32(x)
    elif precision == 'double':
        return np.float64(x)

    
def set_number_of_threads_and_blocks(number_of_beads):
    
    # maxietam number of threads per block : 1024
    
    threads_per_block = 512
    
    if ((number_of_beads // threads_per_block) < 512):
        threads_per_block = 256
    if ((number_of_beads // threads_per_block) < 256):
        threads_per_block = 128
    if ((number_of_beads // threads_per_block) < 128):
        threads_per_block = 64
    if ((number_of_beads // threads_per_block) < 64):
        threads_per_block = 32
    
    num_blocks = (number_of_beads - 1)//threads_per_block + 1
    
    return int(threads_per_block), num_blocks


def no_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):
    """
    Parameters
    ----------
    r_vectors : TYPE
        DESCRIPTION.
    force : TYPE
        DESCRIPTION.
    eta : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    
    """ from P. J. Zuk, E. Wajnryb, K. A. Mizerski and P. Szymczak's paper """
    
    # set the number of threads and blocks
    number_of_beads = len(r_vectors)
    threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_beads)
    
    # reshape arrays
    x = real(np.reshape(r_vectors, number_of_beads * 3).copy())
    f = real(np.reshape(force, number_of_beads * 3).copy())
    
    # allocate gpu memory and copy data (host to device)
    x_gpu = gpuarray.to_gpu(x)
    f_gpu = gpuarray.to_gpu(f)
    u_gpu = gpuarray.empty_like(x_gpu)
    
    
    # get the velocity function kernel
    velocity_pycuda = mod.get_function("velocity_function_no_wall_kernel")
    
    # compute the velocities
    velocity_pycuda(x_gpu, f_gpu, u_gpu, real(eta), np.float32(a), \
                    np.int32(number_of_beads), block = (threads_per_block, 1, 1), grid = (num_blocks, 1))
    
    # copy data from gpu to cpu (device to host)
    u = u_gpu.get()

    return   u.reshape((number_of_beads, 3))
