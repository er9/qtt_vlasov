import pickle
import numpy as np
import axis_map


# def maxwellian(*vs, vth2=1.0, density=1.0, flow=0.0):
#     """ vs: velocity grid for each dimension
# 	    summed together s.t. v^2 = vx^2 + vy^2 + vz^2 for each point
# 	flow: scalar or list (with same # elements as number of velocity dimensions)
#     """
#     v2 = 0.0
#     K = len(vs)
#     for i in range(K):
# 	try:                            v2 += (vs[i]-flow[i])*(vs[i]-flow[i])
# 	except(TypeError,IndexError):   v2 += (vs[i]-flow)   *(vs[i]-flow)
#
#     return density * np.sqrt(1./2/np.pi/vth2)**K * np.exp(-v2/2/vth2)


def maxwellian(vs, vth2=1.0, density=1.0, flow=0.0):
    """ vs: velocity grid for each dimension
            summed together s.t. v^2 = vx^2 + vy^2 + vz^2 for each point
        flow: scalar or list (with same # elements as number of velocity dimensions)
    """
    v2 = (vs-flow)*(vs-flow)
    K = 1
    return density * np.sqrt(1./2/np.pi/vth2)**K * np.exp(-v2/2/vth2)


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

    # return 1. / max(terms)
    return 1./ sum(terms)


def rk4_limit_FourierHermite(m_max, k_max, F_max):
    """ limit dt (estimate) given by rk4 stability constraints:
        1 + z + z^2/2 + z^3/6 + z^4/4 < 1
        for z real, --> z = -2.7853
        for z imag, --> z ~ 1 to ~ 2.75

        calculated assuming in fourier/hermite basis
        df/dt + v * grad_x(f) + q(E)*grad_v(f) = 0
        --> approximate max eigenvalue as sqrt(m_max) * (k_max + E_max)
    """
    dt_max = 1.5 / ( np.sqrt(m_max*2) * (k_max + F_max) )
    # dt_max = 2.75 / (np.sqrt(2 * m_max) * (k_max + F_max))
    # dt_max = 0.723 / (k_max * np.sqrt(m_max/2))     # 0.723 for Adams-Bashforth method
    ### https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/fourierhermite-spectral-representation-for-the-vlasovpoisson-system-in-the-weakly-collisional-limit/ED04894B962D443A41A2EABC7CEC555D
    return dt_max


def rk4_limit_Fourier(k_max, v_max):
    """ limit dt (estimate) given by rk4 stability constraints:
        1 + z + z^2/2 + z^3/6 + z^4/4 < 1
        for z real, --> z = -2.7853
        for z imag, --> z ~ 1 to ~ 2.75

        calculated assuming in fourier/hermite basis
        df/dt + v * grad_x(f) + q(E)*grad_v(f) = 0
        --> approximate max eigenvalue as sqrt(m_max) * (k_max + E_max)
    """
    dt_max = 1.5 / (k_max * v_max)
    return dt_max


def rk4_limit_Hermite(m_max, dx, F_max):
    """ limit dt (estimate) given by rk4 stability constraints:
        1 + z + z^2/2 + z^3/6 + z^4/4 < 1
        for z real, --> z = -2.7853
        for z imag, --> z ~ 1 to ~ 2.75

        calculated assuming in fourier/hermite basis
        df/dt + v * grad_x(f) + q(E)*grad_v(f) = 0
        --> approximate max eigenvalue as sqrt(m_max) * (k_max + E_max)
    """
    dt_max = 1.5 / ( np.sqrt(m_max*2) * (1./dx + F_max) )
    return dt_max



def LB_collision_freq_cfl_limit(dt, dv, vth, vmax, cfl):
    """ solve for collision rate that matches dt, dx, dv specified by the rest of the problem
        von neumann stability analysis (centered FD)
    """
    term1 = vmax / dv * dt
    term2 = 2 * vth ** 2 / dv ** 2 * dt
    freq = cfl / (term1 + term2)
    # print('check cfl', freq, dt, cfl * cfl_limit(((dv, dv ** 2 / 2), (vmax * freq, vth ** 2 * freq))), )

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


def get_maps(mkey: str):
    ax_maps = []
    for mk in mkey:
        if mk == 'F':   # forward
            ax_map = axis_map.BinaryMap()
        elif mk == 'B':     # back
            ax_map = axis_map.FlipBinaryMap()
        elif mk == 'M':
            ax_map = axis_map.MirrorMap()
        elif mk == 'N':
            ax_map = axis_map.FlipMirrorMap()
        else:
            raise KeyError(f'{mk} not a valid map key')
        ax_maps += [ax_map]
    return ax_maps