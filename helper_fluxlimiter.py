from setup_.configs import *

from axis import Axis
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from grid import Grid
from gridTN import GridTN

from local_solvers.terms_3 import Term_Cross
from local_solvers.local_cross_eval import local_cross_evaluator
import local_solvers.helper_tn as helper_tn

def flux_ratio(flux_mps: 'GridTN', deriv_ax: 'Axis') -> 'GridTN':
    D = flux_mps.max_bond()

    shift_mpo = flux_mps.grid.get_shift_mpo({deriv_ax: 1}, ax_boundary_conditions=flux_mps.ax_deriv_configs)
    flux_mps_shift = flux_mps.apply(shift_mpo, compress=True)

    def capped_inv(x: np.ndarray, cap=1e6):
        x_abs = np.abs(x)
        out = np.where(x_abs < 1./ cap, 0, 1./ x)

        # plt.figure()
        # plt.plot(flux_mps.get_data(), 'o-')
        # plt.plot(x ,'x-')
        # plt.figure()
        # plt.plot(out, 'x-')
        # plt.figure()
        # plt.show()
        #

        return out

    def filter(x: np.ndarray, flux_data: np.ndarray):

        correction = -(flux_data - np.roll(flux_data, -1)) / 2
        out = np.where(x < 0, correction, np.zeros(x.shape))
        return out

    ## explicit
    explicit = flux_mps.get_data()
    ratio_explicit = explicit * capped_inv(np.roll(explicit, -1))

    ## semi-expliict
    term1 = Term_Cross(flux_mps_shift.data.copy(), mps_func=capped_inv)
    inv_data = local_cross_evaluator((term1,), nsites=1, max_bond=D)
    flux_mps_shift_inv = flux_mps.create_like(inv_data)

    denom = flux_mps_shift_inv.apply_elemental_multiply_op()
    ratio1 = flux_mps.apply(denom)

    ### fully via cross
    term1c = Term_Cross(flux_mps.data.copy())
    term2c = Term_Cross(flux_mps_shift.data.copy(), mps_func=capped_inv)
    ratio_mps = local_cross_evaluator([term1c, term2c],
                                     max_bond=None, nsites=1,
                                     combine_terms_func=helper_tn.prod_tens)
    ratio = flux_mps.grid.make_gridTN(ratio_mps)

    plt.figure()
    plt.plot(flux_mps.get_data(), 'o-')
    plt.plot(flux_mps_shift.get_data() ,'x-')

    plt.figure()
    plt.plot(capped_inv(np.roll(explicit, -1)), label='explicit inv')
    plt.plot(flux_mps_shift_inv.get_data(), label='cross inv')
    plt.legend()
    plt.title('inv f')
    # plt.show()

    plt.figure()
    plt.plot(ratio_explicit, 'x-', label='explicit')
    plt.plot(ratio1.get_data(), 'o--', label='partial cross')
    plt.plot(ratio.get_data(), 's:', label='full cross')
    plt.legend()
    plt.title('ratio')

    plt.figure()
    plt.plot(filter(ratio_explicit, explicit))
    plt.title('corrections')

    plt.show()

    return ratio


# if __name__ == 'main':

print('here?')

L = 4
npts = 2**L

ax_x = Axis(L, dx=1/npts)
grid_x = Grid1D('X', [ax_x])

# func = lambda x: np.cos(4 * x)
# func = lambda x: np.heaviside(x-0.5, 0.5)
func = lambda x: np.heaviside(x-0.25, 1.0) - 2 * np.heaviside(x-0.5, 1.0) + np.heaviside(x-0.75, 1.0)

f_gtn = grid_x.make_gridTN(func(ax_x.xpts),
                           ax_deriv_configs={ax_x: DerivativeConfiguration(left_bc=BCType.PERIODIC)})


plt.figure()
plt.plot(f_gtn.get_data())
# plt.plot(f_ratio.get_data())
plt.show()


f_ratio = flux_ratio(f_gtn, ax_x)

