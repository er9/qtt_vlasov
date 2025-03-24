import numpy as np
import os, sys, time, glob
sys.path.append('../')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from setup_.helper import cfl_limit


Lx, Ly = 9, 9
npts_x, npts_y = 2**Lx, 2**Ly

permittivity = 1  # 1.0    # permittivity of free space
permeability = 1.0    # permeability of free space
eV = 1.0              # elementary charge or Joule-eV conversion factor

## dipole properties
c = 1/np.sqrt(permittivity*permeability)
dp = 0.2               # dipole length
I0 = 1.0                # current strength
omega = 2 * np.pi * 2
lam = 2 * np.pi / (omega / c)

## box size
Lx = 10.0 * lam
Ly = Lx

x0_vals = np.linspace(-Lx,Lx, npts_x, endpoint=True)
y0_vals = np.linspace(-Ly,Ly, npts_y, endpoint=True)
dx = (x0_vals[1]-x0_vals[0])
dy = (y0_vals[1]-y0_vals[0])
# x1_vals = x0_vals + dx/2
# y1_vals = y0_vals + dy/2

x0_mesh, y0_mesh = np.meshgrid(x0_vals, y0_vals)
# x1_mesh, y1_mesh = np.meshgrid(x1_vals, y1_vals)

## time stepping
T = 5.0
dt = None  # 0.05  # 0.04  # None
cfl = 0.9

cfl_params = ((dx, dy), (c, c))
dt_cfl = cfl_limit( cfl_params )
dt = dt_cfl * cfl if dt is None else dt
num_tsteps = int(T/dt)
print('dt', dt, cfl*1./(c/dx + c/dy), num_tsteps, )
print('1/omega', 1 / omega)
print('dx', dx, 'dy', dy)

""" 
saved values are all of the edge values of each cell (npts-1) cells per dimension
because npts will always be a factor of 2, 
    - if we keep track of the boundaries, Jz centered at zero will be internal to the cell
    - if we only keep track of one of the boundaries (say left boundary), Jz centered at zero 
        will be at npts//2
    - i think we should include both boundaries.
"""

"""
saved values are all at the cores of each cell
cell centers can be at the boundaries.
"""


delta_fct = np.zeros((npts_x, npts_y))
def Jz_t(t):
    delta_fct[npts_x // 2, npts_y // 2] = np.sin(omega * t) * I0
    # print('jz', t, np.sin(omega * t) * I0)
    return delta_fct


def ddx(mat: np.ndarray):
    if isinstance(mat, (int, float)):
        return 0.0

    out = np.zeros((npts_x, npts_y))
    out[1:-1, :] = (mat[2:, :] - mat[:-2, :]) * 0.5
    ## assumes that at x = end + 1, mat is 0;
    ## d/dx at x = 0 is zero.

    # if bc in ['D', 'd', 'dirichlet']:   ## endpoints are zero
    #     out[:-1, :] = mat[1:, :]
    #     out[1:, :] = out[1:, :] - mat[:-1, :]   ## out[1:, :]
    # elif bc in ['N', 'n', 'neumann']:   ## endpoints have zero gradient
    #     out[1:, :] = mat[1:, :] - mat[:-1, :]
    # else:
    #     raise TypeError
    return out

def ddy(mat: np.ndarray):
    if isinstance(mat, (int, float)):
        return 0.0

    out = np.zeros((npts_x, npts_y))
    out[:, 1:-1] = (mat[:, 2:] - mat[:, :-2]) * 0.5
    return out

def d2dx2(mat: np.ndarray):
    if isinstance(mat, (int, float)):
        return 0.0

    out = mat * -2
    out[:-1, :] += mat[1:, :]
    out[1:, :] += mat[:-1, :]
    return out

def d2dy2(mat: np.ndarray):
    if isinstance(mat, (int, float)):
        return 0.0

    out = mat * -2
    out[:, :-1] += mat[:, 1:]
    out[:, 1:] += mat[:, :-1]
    return out

def avg_x(mat: np.ndarray):
    if isinstance(mat, (int, float)):
        return mat

    out = mat * 0.5
    out[:-1, :] += mat[1:,:] * 0.25
    out[1:, :] += mat[:-1, :] * 0.25
    return out

def avg_y(mat: np.ndarray):
    if isinstance(mat, (int, float)):
        return mat

    out = mat * 0.5
    out[:, :-1] += mat[:, 1:] * 0.25
    out[:, 1:] += mat[:, :-1] * 0.25
    return out


def calc_deriv(Bx: np.ndarray, By: np.ndarray, Ez: np.ndarray, Jz: np.ndarray):
    """ dEz/dt - c^2 (d/dx By - d/dy Bx) - c/2 (dx d2/dx2 + dy d2/dy2) Ez
                + 1/2 (Jz_(i+1/2,j) + Jz_(i,j+1/2)) = 0
        dBx/dt + (d/dy Ez - d/dz Ey) - c/2 (dy d2/dy2 + dz d2/dz2) Bx
                - 1/2 (dy d/dy Jz) = 0
        dBy/dt + (d/dz Ex - d/dx Ez) - c/2 (dx d2/dx2 + dz d2/dz2) By
                - 1/2 (dx d/dx Jz) = 0
    """
    dEzdt = c**2 * ddx(By) / dx - c**2 * ddy(Bx) / dy
    dBxdt = - ddy(Ez) / dy
    dBydt = + ddx(Ez) / dx

    # plt.figure()
    # plt.imshow(Ez)
    # plt.title('Ez')
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(d2dy2(Ez) / dy)
    # plt.title('d2dy2 Ez')
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(d2dx2(Ez) / dx)
    # plt.title('d2dx2 Ez')
    # plt.colorbar()
    # plt.show()

    # plt.fia
    do_upwind = True
    if do_upwind:
        dEzdt = dEzdt \
                    + c/2 * (d2dx2(Ez) / dx + d2dy2(Ez) / dy) \
                    - 0.5 * (avg_x(Jz) + avg_y(Jz)) / permittivity
                    # - Jz / permittivity

        dBxdt = dBxdt + c/2 * (d2dy2(Bx) / dy) - 1 / 2. / c * ddy(Jz)
        dBydt = dBydt + c/2 * (d2dx2(By) / dx) + 1 / 2. / c * ddx(Jz)
    else:
        dEzdt = dEzdt - Jz / permittivity

    return dBxdt, dBydt, dEzdt


# ### FDTD ###
# def euler(dt, Bx, By, Ez, Jz, deriv=None):
#     print('FDTD')
#     if deriv is None:
#         deriv_Bx, deriv_By, deriv_Ez = calc_deriv(Bx, By, Ez, Jz)
#     else:
#         deriv_Bx, deriv_By, deriv_Ez = deriv
#
#     new_Bx = Bx + deriv_Bx * dt
#     new_By = By + deriv_By * dt
#     # new_Ez = Ez + deriv_Ez * dt
#
#     deriv_Bx, deriv_By, deriv_Ez = calc_deriv(new_Bx, new_By, Ez, Jz)
#     new_Ez = Ez + deriv_Ez * dt
#     # new_Bx = Bx + deriv_Bx * dt
#     # new_By = By + deriv_By * dt
#
#     return new_Bx, new_By, new_Ez


# ### FV w/ time-splitting of source ###
# def euler(dt, Bx, By, Ez, Jz, deriv=None):
#     if deriv is None:
#         deriv_Bx, deriv_By, deriv_Ez = calc_deriv(Bx, By, Ez, 0)  # Jz)
#     else:
#         deriv_Bx, deriv_By, deriv_Ez = deriv
#
#     Ez = Ez - Jz / permittivity * dt / 2
#     new_Bx = Bx + deriv_Bx * dt
#     new_By = By + deriv_By * dt
#     new_Ez = Ez + deriv_Ez * dt
#     new_Ez = new_Ez - Jz / permittivity * dt / 2
#
#     return new_Bx, new_By, new_Ez

### FV w/ source ###
def euler(dt, Bx, By, Ez, Jz, deriv=None):
    if deriv is None:
        deriv_Bx, deriv_By, deriv_Ez = calc_deriv(Bx, By, Ez, Jz)
    else:
        deriv_Bx, deriv_By, deriv_Ez = deriv

    new_Bx = Bx + deriv_Bx * dt
    new_By = By + deriv_By * dt
    new_Ez = Ez + deriv_Ez * dt

    return new_Bx, new_By, new_Ez




Bx_ = np.zeros((npts_x, npts_y))
By_ = np.zeros((npts_x, npts_y))
Ez_ = np.zeros((npts_x, npts_y))

nt = 0
while nt < num_tsteps:

    print('nt', nt, 'dt_', dt, 't', nt*dt)
    Jz_ = Jz_t(nt*dt + dt/2)

    Bx_, By_, Ez_ = euler(dt, Bx_, By_, Ez_, Jz_)
    nt += 1

    if nt%10 == 0:
        plt.figure()
        plt.imshow(Ez_)
        plt.title('Ez')
        plt.colorbar()

        plt.figure()
        plt.imshow(Bx_)
        plt.title('Bx')
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.plot(Ez_)
        plt.title('Ez')

        plt.figure()
        plt.plot(Bx_)
        plt.title('Bx')
        plt.show()
