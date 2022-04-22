import numpy as np
from basis import Basis

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axis import Axis


class HermiteBasis(Basis):
    """ physicist Hermite polynomials
                  H_m(x) = (-1)^m exp(x^2) d^n/dx^n exp(-x^2)
                  orthogonal (int dx H_m(x) H_n(x) = delta_m,n)
                  x * H_m(x)  = m * H_(m-1)(x) + 1/2 * H_(m+1)(x)
                  d/dx H_m(x) = 2*m * H_(m-1)(x)
    """
    @staticmethod
    def get_realspace_1D(coeffs,x_window,npts,x_offset=0.,x_scale=1.):
        """ coeffs:  coefficients of Hermite polynomials (order m = 0 to M-1)
            x_window:  x0,xL specifying domain of interest
            x_offset:  offset of center of Hermite polynomials
            x_scale:   scaling of x in Hermite polynomials     (x <- x_scale*x + x_offset)
            note: numpy fct is physicists Hermite polynomials (no normalization)
        """
        x0,xL = x_window
        domain = ((x0-x_offset)/x_scale, (xL-x_offset)/x_scale)
        hermite_poly = np.polynomial.hermite.Hermite(coeffs,domain,x_window)
        return hermite_poly.linspace(npts)

    @staticmethod
    def elemental_multiply(tn1,tn2,inplace=False,**compress_opts):
        raise NotImplementedError

    @staticmethod
    def build_xmulitply_mpo(tn_grid, ax:'Axis', offset=0.0, scale=1.0, split_opts=None):
        """ v * H_m(v)
            NOTE: also include hermite bsais offset+scaling here, ie. (x-x0)*H_m(x-x0) + x0*H_m(x-x0)
        """
        grid = tn_grid.get_subgrid(ax)
        q, L = grid.q, grid.L

        ms = np.arange(1,q**L)
        mat = np.diag(ms,k=1) + np.diag(1./2*np.ones(q**L-1),k=-1)   # m->m-1, m->m+1
        mpo = grid.map_operator_to_mpo(mat, split_opts=split_opts)
        return mpo

    @staticmethod
    def build_firstderivative_mpo(tn_grid,ax):
        """ dH_{m}/dv = 2*m*H_{m-1}
        """
        grid = tn_grid.get_subgrid(ax)
        q, L = grid.q, grid.L
        compress_opts = grid.get_compress_opts(-1)

        ms = np.arange(1,q**L)
        mat = np.diag(2*ms,k=1)   # m->m-1
        mpo = grid.map_operator_to_mpo(mat, None),
        return mpo

    @staticmethod
    def build_secondderivative_mpo(tn_grid,ax):
        grid = tn_grid.get_subgrid(ax)
        q, L = grid.q, grid.L
        compress_opts = grid.get_compress_opts(-1)

        ms = np.arange(1,q**L)
        mat = np.diag(np.sqrt(ms/2),k=1) - np.diag(np.sqrt(ms/2),k=-1)   # m->m-1, m->m+1
        mat = np.dot(mat,mat)
        mpo = grid.map_operator_to_mpo(mat, None),
        return mpo


class HermiteGaussianBasis:
    """ Hermite-Gaussian polynomials
                  H_m(x) = (-1)^m / (2^m m! sqrt(pi)) exp(x^2/2) d^n/dx^n exp(-x^2)
                  orthogonal (int dx H_m(x) H_n(x) = delta_m,n)
                  x * H_m(x)  = np.sqrt(m/2) * H_(m-1)(x) + np.sqrt((m+1)/2) * H_(m+1)(x)
                  d/dx H_m(x) = np.sqrt(m/2) * H_(m-1)(x) - np.sqrt((m+1)/2) * H_(m+1)(x)
    """
    @staticmethod
    def get_realspace_1D(coeffs,x_window,npts,x_offset=0.,x_scale=1.):
        """ coeffs:  coefficients of Hermite polynomials
            x_window:  x0,xL specifying domain of interest
            x_offset:  offset of center of Hermite polynomials
            x_scale:   scaling of x in Hermite polynomials     (x <- x_scale*x + x_offset)
            note: numpy fct is physicists Hermite polynomials (no normalization)
        """
        x0,xL = x_window
        domain = ((x0-x_offset)/x_scale, (xL-x_offset)/x_scale)

        ms = np.araange(len(coeffs))
        coeffs *= 1./np.sqrt(2**ms * np.math.factorial(ms) * np.sqrt(np.pi))

        hermite_poly = np.polynomial.hermite.Hermite(coeffs,domain,x_window)
        xs = np.linspace(x0,xL,npts)
        hermite_poly *= np.exp(-xs**2/2)

        return hermite_poly.linspace(npts)

    @staticmethod
    def elemental_multiply(tn1,tn2,inplace=False,**compress_opts):
        raise NotImplementedError

    @staticmethod
    def build_mpo_xmulitply(tn_grid,ax):
        """ x * H_m(x)
        """
        grid = tn_grid.get_subgrid(ax)
        q, L = grid.q, grid.L
        compress_opts = grid.get_compress_opts(-1)

        ms = np.arange(1,q**L)
        mat = np.diag(np.sqrt(ms/2),k=1) + np.diag(np.sqrt(ms/2),k=-1)   # m->m-1, m->m+1
        mpo = grid.map_operator_to_mpo(mat, None),
        return mpo


    @staticmethod
    def build_mpo_firstderivative(tn_grid,ax):
        """ dH_{m}/dv = 2*m*H_{m-1}
        """
        grid = tn_grid.get_subgrid(ax)
        q, L = grid.q, grid.L
        compress_opts = grid.get_compress_opts(-1)

        ms = np.arange(1,q**L)
        mat = np.diag(np.sqrt(ms/2),k=1) - np.diag(np.sqrt(ms/2),k=-1)   # m->m-1, m->m+1
        mpo = grid.map_operator_to_mpo(mat, None),
        return mpo

    @staticmethod
    def build_mpo_secondderivative(tn_grid,ax):
        grid = tn_grid.get_subgrid(ax)
        q, L = grid.q, grid.L
        compress_opts = grid.get_compress_opts(-1)

        ms = np.arange(1,q**L)
        mat = np.diag(np.sqrt(ms/2),k=1) - np.diag(np.sqrt(ms/2),k=-1)   # m->m-1, m->m+1
        mat = np.dot(mat,mat)
        mpo = grid.map_operator_to_mpo(mat, None),
        return mpo
