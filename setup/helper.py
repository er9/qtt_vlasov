import pickle
import numpy as np


def cfl_limit(*pde_params):
    """ compute CFL time-step limit given sets of grid params
        eq_params:  list of nested tuples:
                    [((dx, dv), (vmax, Fmax)), ... ]
        result: 1./ max( (vmax/dx + Fmax/dv, ...) )
    """
    terms = []
    for param in pde_params:
        dxs, Fs = param
        terms += [np.sum([F / dx for F, dx in zip(Fs, dxs)])]

    return 1. / max(terms)


def LB_collision_freq_cfl_limit(dt, dv, vth, vmax, cfl):
    """ solve for collision rate that matches dt, dx, dv specified by the rest of the problem
        von neumann stability analysis (centered FD)
    """
    term1 = vmax / dv * dt
    term2 = 2 * vth ** 2 / dv ** 2 * dt
    freq = cfl / (term1 + term2)
    print('check cfl', freq, dt, cfl * cfl_limit(((dv, dv ** 2 / 2), (vmax * freq, vth ** 2 * freq))), )

    return freq


def optional_load(filename):
    """ try loading the data else return None
    """
    try:
        return pickle.load(open(filename, 'rb'))
    except IOError:
        return None


def optional_dump(obj, filename):
    """ try loading the MPS representing some state
    """
    if obj is not None:
        pickle.dump(obj, open(filename, 'wb'))
    else:
        pass